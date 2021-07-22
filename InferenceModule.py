# -*- coding: UTF-8 -*-

import os
import shutil
import pickle
import csv
import zhconv
import networkx as nx
import numpy as np
import subprocess
from ConceptNetModule import ConceptNetModule
from BayesianNetworkModule import BayesianNetworkModule
from snownlp import SnowNLP


class InferenceModule:
    def __init__(self):
        self.code2node = dict()
        self.table_phrase_type = dict()
        self.negative_emotion = dict()
        self.node2stressor = dict()
        self.code_observed_nodes = None
        self.inference_result = None
        self.type_graph = None

    def initialize(self, type_graph, code2node):
        # with open('./LocalTempData/code2node_{}.pickle'.format(type_graph), 'rb') as file:
        # 	self.code2node = pickle.load(file)

        with open("./LocalCacheFile/phraseTypeCache.pickle", "rb") as file:
            self.table_phrase_type = pickle.load(file)

        with open("./LocalCacheFile/stressorSet.pickle", "rb") as file:
            self.node2stressor = pickle.load(file)

        with open("./LocalCacheFile/negativeEmotion.pickle", "rb") as file:
            self.negative_emotion = pickle.load(file)

        self.code2node = code2node
        self.code_observed_nodes = self.load_csv_file(
            "./LocalTempData/codeObservedNodes_{}.csv".format(type_graph)
        )
        self.inference_result = self.load_csv_file(
            "./LocalTempData/gibbsResults_{}.csv".format(type_graph)
        )
        self.type_graph = type_graph

    def load_csv_file(self, file_path):
        info_seg = None

        if not os.path.isfile(file_path):
            return None

        with open(file_path, "r") as file:
            csv_info = csv.reader(file, delimiter=",", quotechar="|")
            for row_info in csv_info:
                info_seg = row_info

        return info_seg

    def get_sub_weight(self, weight):
        if weight > 1:
            return 1
        else:
            return weight

    def normalize_stressor_weight(self, stressor_dict):
        sum_weight = 0
        for key in stressor_dict:
            sum_weight += stressor_dict[key]

        for key in stressor_dict:
            stressor_dict[key] = stressor_dict[key] / sum_weight

        return stressor_dict

    def sort_inferred_result(self):
        if self.type_graph == "Stressor":

            if self.inference_result == None:
                return None

            result = [
                (self.code2node[key], float(value))
                for key, value in enumerate(self.inference_result)
            ]
            print("Stressor Result: ", result)
            # sub_weight: sub_weight['PH1'] = {weight = 2, num = 3}
            # stressor_weight: stressor_weight['Frustration'] = {weight = 1, num = 2}
            # Final_weight = 1 / 2 = 0.5
            sub_weight = dict()
            stressor_weight = dict()
            for node_result in result:
                for stressor in self.node2stressor:
                    for key, value in self.node2stressor[stressor].items():
                        if node_result[0] in value:
                            if key in sub_weight:
                                sub_weight[key]["weight"] += node_result[1]
                                sub_weight[key]["num"] += 1
                            else:
                                sub_weight[key] = {"weight": node_result[1], "num": 1}

            # Average each sub_weight with the number of node
            # sub_weight: sub_weight['PH1'] = 0.67
            for node in sub_weight:
                sub_weight[node] = sub_weight[node]["weight"] / sub_weight[node]["num"]

            for key in sub_weight:
                for stressor in self.node2stressor:
                    if key in self.node2stressor[stressor]:
                        if stressor in stressor_weight:
                            weight = self.get_sub_weight(sub_weight[key])
                            stressor_weight[stressor]["weight"] += weight
                            stressor_weight[stressor]["num"] += 1
                        else:
                            weight = self.get_sub_weight(sub_weight[key])
                            stressor_weight[stressor] = {"weight": weight, "num": 1}

            # Average each stressor_weight with the number of stressor
            # stressor_weight: stressor_weight['Frustration'] = 0.5
            for stressor in stressor_weight:
                stressor_weight[stressor] = (
                    stressor_weight[stressor]["weight"]
                    / stressor_weight[stressor]["num"]
                )

            stressor_weight = self.normalize_stressor_weight(stressor_weight)

            return stressor_weight
        else:

            if self.inference_result == None:
                return None, None

            # Skip the inferred result of observed nodes
            result = [
                (self.code2node[key], float(value))
                for key, value in enumerate(self.inference_result)
                if not str(key) in self.code_observed_nodes
            ]

            feeling_list = list()
            advice_list = list()
            for node_result in result:
                # Skip the node which is not in the phrase_type table
                if not node_result[0] in self.table_phrase_type:
                    continue
                if self.table_phrase_type[node_result[0]] == "emotion":
                    if node_result[0] in self.negative_emotion:
                        feeling_list.append(node_result)
                elif self.table_phrase_type[node_result[0]] == "verbPhrase":
                    advice_list.append(node_result)

            sorted_feeling = sorted(feeling_list, key=lambda x: x[1], reverse=True)
            print("Feeling Result: ", sorted_feeling)
            sorted_advice = sorted(advice_list, key=lambda x: x[1], reverse=True)

            return sorted_feeling, sorted_advice


def main():
    # Initialize
    shutil.rmtree("./LocalTempData")
    os.mkdir("./LocalTempData")

    CN = ConceptNetModule()
    BN = BayesianNetworkModule()
    IM = InferenceModule()

    # Query the graph of observed node (Feeling&Advice => FA; Stressor => S)
    topic_sentence = ["沒時間", "考試考不好"]
    CN.initialize("Query")
    FA_observation, FA_graph = CN.query_graph_for_feeling_advice(topic_sentence)
    S_observation, S_graph = CN.queried_graph_for_stressor(topic_sentence)

    # Create Bayesina Network for applying Gibbs Sampling (C++)
    BN.initialize("Feeling&Advice", FA_observation, FA_graph)
    FA_code2node = BN.build_bayesian_network()
    BN.initialize("Stressor", S_observation, S_graph)
    S_code2node = BN.build_bayesian_network()

    # Use Gibbs Sampling to find marginal probability
    subprocess.call(["./infer", "Feeling&Advice"])
    subprocess.call(["./infer", "Stressor"])

    # Show the inferred answer
    IM.initialize("Feeling&Advice", FA_code2node)
    feeling, advice = IM.sort_inferred_result()
    IM.initialize("Stressor", S_code2node)
    stressor = IM.sort_inferred_result()

    print("Feeling: ", feeling)
    print("Advice: ", advice)
    print("Stressor: ", stressor)


if __name__ == "__main__":
    main()
