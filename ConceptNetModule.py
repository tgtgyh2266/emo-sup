# -*- coding: UTF-8 -*-

import sys
import pickle
import tables
import zhconv
import pickle
import requests
import time
import networkx as nx


class ConceptNetModule:
    def __init__(self):
        self.conceptnet_nodes = dict()
        self.conceptnet_cached = dict()
        self.table_phrase_type = dict()
        # self.stressor_categories = set()
        self.DG_feeling_advice = nx.DiGraph()
        self.DG_init_stressor = nx.DiGraph()
        self.DG_stressor = nx.DiGraph()

    def initialize(self, style):

        if style == "Build":
            # Load the chinese concept nodes from ConceptNet
            # Those nodes are saved in conceptNetNodes.pickle and all words are in Traditional Chinese
            with open("./LocalCacheFile/conceptNetNodes.pickle", "rb") as file:
                self.conceptnet_nodes = pickle.load(file)
        elif style == "Query":
            # Load the local conceptnet from disk
            with open("./LocalCacheFile/conceptNetCache.pickle", "rb") as file:
                self.conceptnet_cached = pickle.load(file)

            # Load the local stressor graph
            with open(
                "./LocalCacheFile/initialConceptGraph_Stressor.pickle", "rb"
            ) as file:
                self.DG_init_stressor = pickle.load(file)

            with open("./LocalCacheFile/phraseTypeCache.pickle", "rb") as file:
                self.table_phrase_type = pickle.load(file)

        # Initialize the directed graph
        self.DG_feeling_advice = nx.DiGraph()
        self.DG_stressor = nx.DiGraph()

    def cache_all_chinese_conceptnet_to_disk(self):
        # Cache the conceptnet from web to local (disk)
        # Temp: end in 32,000
        cached_conceptnet = dict()

        # Cache: All
        for key in self.conceptnet_nodes:
            start = time.time()
            obj = requests.get(
                "http://api.conceptnet.io/query?start=/c/zh/{}&offset=0&limit=2000".format(
                    self.conceptnet_nodes[key]
                )
            ).json()

            if "edges" in obj:
                if len(obj["edges"]) > 0:
                    cached_concept = dict()
                    cached_concept["end"] = [
                        edge["end"]["label"] for edge in obj["edges"]
                    ]
                    cached_concept["rel"] = [
                        edge["rel"]["label"] for edge in obj["edges"]
                    ]
                    cached_concept["weight"] = [edge["weight"] for edge in obj["edges"]]
                    cached_conceptnet[self.conceptnet_nodes[key]] = cached_concept

                    print(
                        "Processing - Index: {}, Node: {:^8}, QueryTime: {}".format(
                            key, self.conceptnet_nodes[key], time.time() - start
                        ),
                        end="\r",
                    )

            # Save 1000 cached concept in one file
            if key % 1000 == 0 and key != 0:
                with open(
                    "./LocalCacheFile/conceptNetCache_{}.pickle".format(key), "wb"
                ) as file:
                    pickle.dump(cached_conceptnet, file)
                cached_conceptnet = dict()

        # Cache: StartIndex ~ End
        # for key in range(47001, len(self.conceptnet_nodes)):
        # 	start = time.time()
        # 	obj = requests.get('http://api.conceptnet.io/query?start=/c/zh/{}&offset=0&limit=2000'.format(self.conceptnet_nodes[key])).json()
        # 	if 'edges' in obj:
        # 		if len(obj['edges']) > 0:
        # 			cached_concept = dict()
        # 			cached_concept['end'] = [edge['end']['label'] for edge in obj['edges']]
        # 			cached_concept['rel'] = [edge['rel']['label'] for edge in obj['edges']]
        # 			cached_concept['weight'] = [edge['weight'] for edge in obj['edges']]
        # 			cached_conceptnet[self.conceptnet_nodes[key]] = cached_concept

        # 			print('Processing - Index: {}, Node: {:^8}, QueryTime: {}'.format(key, self.conceptnet_nodes[key], time.time() - start), end='\r')

        # 	# Save 1000 cached concept in one file
        # 	if (key % 1000 == 0 and key != 0) or key == 47648:
        # 		with open ('conceptNetCache_{}.pickle'.format(key), 'wb') as file:
        # 			pickle.dump(cached_conceptnet, file)
        # 		cached_conceptnet = dict()

        print("\nCache Done!")

    def query_graph_for_feeling_advice(self, concept_list):
        # This function is used to query concepts from the local ConceptNet
        # It contains concepts and causal relational edges (which are applied for inferring feeling and advices)

        # print(concept_list)

        # The concerned edge relations dictionary:
        # A Causes B; A CauseDesire B; A Desires B;
        concerned_relations = ["Causes", "CausesDesire", "Desires"]

        observed_nodes = set()

        # For ConceptNet which is dict_type
        # To query the concepts one by one, and build the graph
        for concept in concept_list:
            sub_conceptnet = dict()
            if concept in self.conceptnet_cached:
                # Retrieve the local cache
                sub_conceptnet = self.conceptnet_cached[concept]
            else:
                # Query from Web API
                for query_edge in concerned_relations:
                    obj = requests.get(
                        "http://api.conceptnet.io/query?start=/c/zh/{}".format(
                            concept, query_edge
                        )
                    ).json()
                    sub_conceptnet["end"] = [
                        edge["end"]["label"] for edge in obj["edges"]
                    ]
                    sub_conceptnet["rel"] = [
                        edge["rel"]["label"] for edge in obj["edges"]
                    ]
                    sub_conceptnet["weight"] = [edge["weight"] for edge in obj["edges"]]

            # concept_weight: The weight of word in the sentence or the concept's overlap rate in the sentence
            # (overlap rate)For example: the sentence is ABCDEFGH, the concept is ABCD then the overlap rate is
            # len(ABCD) / len(ABCDEFGH) = 4 / 8 = 0.5, so all edge weight of this concept will multiple by 0.5
            concept_weight = 1 * concept_list[concept]

            # Build graph of feeling and advices (only keep the concerned_relations)
            # First, find the index of sub_conceptnet (the value of 'rel' match concerned_relations)
            index_list = [
                index
                for index, relation in enumerate(sub_conceptnet["rel"])
                if relation in concerned_relations
            ]

            # Second, set start node, end node, weight
            # There may have more than one relation between same start node and end node-> sum all weights
            start_node = concept

            # Add the observed node, the probability of this node will be set as 1.0 in the Gibbs Sampling (Bayesian Network)
            # Make sure that this concept is on the conceptNet graph (subgraph)
            observed_nodes.add(start_node) if len(index_list) > 0 else observed_nodes
            # print('Obseration: ', observed_nodes)

            for index in index_list:
                end_node = sub_conceptnet["end"][index]
                weight = sub_conceptnet["weight"][index] * concept_weight

                # Add the node, edge and the corresponding weight to the graph
                if (not end_node in observed_nodes) and (
                    end_node in self.table_phrase_type
                ):
                    if self.table_phrase_type[end_node] in ["emotion", "verbPhrase"]:
                        self.DG_feeling_advice = self.add_edge_for_graph(
                            self.DG_feeling_advice, start_node, end_node, weight
                        )
                    else:
                        self.DG_feeling_advice.add_node(start_node)
                        # print('Inner Start node: {}, End node: {}'.format(start_node, end_node))
                else:
                    self.DG_feeling_advice.add_node(start_node)
                    # print('Start node: {}, End node: {}'.format(start_node, end_node))
        return observed_nodes, self.DG_feeling_advice

    def build_graph_for_stressor(self):
        # This function is used to build the graph for 5(6) stressors
        # Physcial, mental, (emotional) demands, time pressure, isolation, frustration
        # Key: stressor; Value: the node which has relations with stressor
        stressors = dict()
        stressors["physical_demands"] = ["睡眠不足", "身體不好", "受傷"]
        stressors["mental_demands"] = ["專心", "專注"]
        stressors["time_pressure"] = ["時間不夠"]
        stressors["isolation"] = ["獨自一人", "孤單", "一個人"]
        stressors["frustration"] = ["挫折", "失落", "失望", "失敗"]

        stressors["sadness"] = ["傷心","難過"]
        stressors["anger"] = ["憤怒","生氣","不爽"]
        stressors["fear"] = ["害怕","恐懼"]

        # What relations should concern
        end_concerned_relations = [
            "Causes",
            "CausesDesire",
            "Desires",
            "HasSubevent",
            "HasFirstSubevent",
        ]

        # Add the node which has relations with stressor into the graph
        for key in stressors:
            for source_node in stressors[key]:
                # Build the edge between source node and stressor
                self.DG_init_stressor.add_edge(source_node, key, weight=1.0)
                # Find node which is relation match the concerned relations and its neighbor matches the source node
                # For end relations mean that source node is end node
                for concept in self.conceptnet_cached:
                    for index, relation in enumerate(
                        self.conceptnet_cached[concept]["rel"]
                    ):
                        if (
                            relation in end_concerned_relations
                            and self.conceptnet_cached[concept]["end"][index]
                            == source_node
                        ):
                            start_node = concept
                            end_node = source_node
                            weight = self.conceptnet_cached[concept]["weight"][index]

                            # Add the node, edge and the corresponding weight to the graph
                            self.DG_init_stressor = self.add_edge_for_graph(
                                self.DG_init_stressor, start_node, end_node, weight
                            )

        # Save the init graph for stressor
        with open("./LocalCacheFile/initialConceptGraph_Stressor.pickle", "wb") as file:
            pickle.dump(self.DG_init_stressor, file)

    def queried_graph_for_stressor(self, concept_list):
        # Record the observed nodes
        observed_nodes = set()

        # This function is used to queried subgraph of stressor according to the concept_list
        for concept in concept_list:
            concept_weight = 1 * concept_list[concept]
            # Get the subgraph of concept (User input) DG_init_stressor only has 3 depth
            if self.DG_init_stressor.has_node(concept):
                observed_nodes.add(concept)
                self.DG_stressor.add_node(concept)

                # Find the child node of the concept and reset the weight
                for child in self.DG_init_stressor.successors(concept):
                    weight = (
                        self.DG_init_stressor[concept][child]["weight"] * concept_weight
                    )
                    self.DG_stressor = self.add_edge_for_graph(
                        self.DG_stressor, concept, child, weight
                    )

        # print(self.DG_stressor.edges())
        return observed_nodes, self.DG_stressor

    def add_edge_for_graph(self, graph, start_node, end_node, weight):
        if graph.has_edge(start_node, end_node):
            graph[start_node][end_node]["weight"] += weight
        else:
            graph.add_edge(start_node, end_node, weight=weight)

        return graph

    def save_observed_nodes(self, type_graph, observed_nodes):
        with open(
            "./LocalTempData/observedNodes_{}.pickle".format(type_graph), "wb"
        ) as file:
            pickle.dump(observed_nodes, file)

    def save_queried_concept_graph(self, type_graph, concept_graph):
        # This function is used to save the networkx graph of specific concept_graph
        with open(
            "./LocalTempData/queriedConceptGraph_{}.pickle".format(type_graph), "wb"
        ) as file:
            pickle.dump(concept_graph, file)


def main():
    print("ConceptNetModule")

    CN = ConceptNetModule()
    CN.initialize("Query")
    # CN.cache_all_chinese_conceptnet_to_disk()


if __name__ == "__main__":
    main()
