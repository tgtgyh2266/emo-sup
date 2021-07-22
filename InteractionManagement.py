# -*- coding: UTF-8 -*-

import os
import sys
import shutil
import subprocess
import zhconv
import time

import numpy as np

from NaturalLanguageUnderstanding import NaturalLanguageUnderstanding
from ConceptNetModule import ConceptNetModule
from BayesianNetworkModule import BayesianNetworkModule
from InferenceModule import InferenceModule
from DecisionModule import DecisionModule
from GeneralChat import GeneralChat
from EmotionalSupportChat import EmotionalSupportChat

# from Swinger import Swinger

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class InteractionManagement:
    def __init__(self, trait):
        self.NLU = NaturalLanguageUnderstanding()
        self.CN = ConceptNetModule()
        self.CN.initialize("Query")
        self.BN = BayesianNetworkModule()
        self.IFM = InferenceModule()
        self.DM = DecisionModule()
        self.GC = GeneralChat()
        self.ESC = EmotionalSupportChat()
        # self.swinger = Swinger()
        # self.swinger.load('LogisticRegression')
        self.trait_vector = self.to_categorical(trait, 5)
        self.stressor_vector = None
        self.current_support = list()
        self.advice_loop = False
        self.temp_sentence = None
        self.current_stressor = None

    def to_categorical(self, input_class, num_classes):
        res = np.zeros(num_classes, dtype=float)
        res[input_class] = 1
        return res

    def create_stressor_vector(self, stressor):
        # Create stressor vector
        stressor_vector = np.zeros(6, dtype=float)

        for key, value in stressor.items():
            if key == "Physiology":
                stressor_vector[0] = value
            elif key == "Pressure":
                stressor_vector[1] = value
            elif key == "Frustration":
                stressor_vector[2] = value
            elif key == "Conflict":
                stressor_vector[3] = value
            elif key == "Change":
                stressor_vector[4] = value
            elif key == "Isolation":
                stressor_vector[5] = value

        if sum(stressor_vector) == 0:
            return None
        else:
            return stressor_vector

    def emotional_support_system(self, sentence, current_id):
        if "拜拜" in sentence or "再见" in sentence:
            self.current_stressor = None
            self.current_support = list()
            return "再見，希望你能順利度過困境，下次難過的時候，隨時歡迎你繼續找我聊天"
        # Initialize
        shutil.rmtree("./LocalTempData")
        os.mkdir("./LocalTempData")

        # Use the NLU to capture key words of sentence
        concept_keys = self.NLU.capture_key_words(sentence)
        print("==============================================")
        print("Keyword Capture: ", concept_keys)
        print("==============================================")

        # Negative sentence
        # if self.swinger.swing(zhconv.convert(sentence, 'zh-tw')) == 'neg':
        # Query the graph of observed node (Feeling&Advice => FA; Stressor => S)
        self.CN.initialize("Query")
        FA_observation, FA_graph = self.CN.query_graph_for_feeling_advice(concept_keys)
        S_observation, S_graph = self.CN.queried_graph_for_stressor(concept_keys)
        # print(S_observation, S_graph.edges(data='weight'))
        # print(FA_observation, FA_graph.edges(data='weight'))

        # Create Bayesina Network for applying Gibbs Sampling (C++)
        self.BN.initialize("Stressor", S_observation, S_graph)
        S_code2node = self.BN.build_bayesian_network()

        # Use Gibbs Sampling to find marginal probability
        subprocess.call(["./infer", "Stressor"])

        # Infer stressor answer
        self.IFM.initialize("Stressor", S_code2node)
        stressor_category = self.IFM.sort_inferred_result()

        # Infer feeling answer
        if list(FA_graph.nodes()):
            self.BN.initialize("Feeling&Advice", FA_observation, FA_graph)
            FA_code2node = self.BN.build_bayesian_network()

            subprocess.call(["./infer", "Feeling&Advice"])

            self.IFM.initialize("Feeling&Advice", FA_code2node)
            feeling, advice = self.IFM.sort_inferred_result()
        else:
            feeling, advice = [], []

        print("Current State: ", S_observation)
        print("==============================================")

        # if stressor_category and self.ESC.checkrepository(S_observation):
        if stressor_category and (self.ESC.checkrepository(S_observation) or self.ESC.checkrepository(concept_keys)): #modified 2021/06/18, to detect keywords directly
            self.stressor_vector = self.create_stressor_vector(stressor_category)
            input_vector = np.expand_dims(
                np.concatenate([self.stressor_vector, self.trait_vector], axis=-1),
                axis=0,
            )

            emotional_support = self.DM.model_predict(input_vector)

            # print("==============================================")
            # print("Stressor: ", stressor_category)
            # print("==============================================")
            # print("Emotional Support: ", emotional_support)
            # print("==============================================")
            # if 'empathy' in emotional_support:
            # 	print("Feeling: ", feeling)
            # print("==============================================")

            # Alternate chat
            # if self.current_stressor:
            # 	sentence = self.current_stressor + '讓' + sentence

            # general_response = list(self.NLU.splitSentence(self.GC.respond(sentence)))
            # # general_response = ''.join(general_response[0]) if len(general_response) > 3 else ''.join(general_response[0:3])
            # general_response = "".join(general_response[0:])

            general_response = self.GC.respond(sentence)

            (
                es_response,
                current_stressor,
                current_support,
                special_mark,
            ) = self.ESC.respond(
                sentence,
                concept_keys,
                feeling,
                emotional_support,
                current_id,
                self.current_stressor,
            )

            print("Test: ",self.current_stressor, current_stressor, " : ", special_mark)

            if special_mark:
                self.current_stressor = current_stressor
                self.current_support = current_support
                if es_response:
                    self.GC.respond("No!")
                    response = es_response
                else:
                    response = general_response
                    # added by Steven
                    # do it really help?
                    # response = es_response
            elif (not self.current_stressor == current_stressor) and current_stressor:
                print('es_response: ', es_response)
                self.current_stressor = current_stressor
                self.current_support = current_support
                if es_response:
                    self.GC.respond("No!")
                    response = es_response
                else:
                    response = general_response
            else:
                self.current_stressor = list()
                response = general_response
            # response = es_response if es_response else general_response
            # self.GC.respond(response)
            print(
                "Response in Emotional Support System (With Stressor): ",
                response,
                ";",
                current_stressor,
            )
        else:

            print("Current Stressor: ", self.current_stressor)
            print("==============================================")

            if self.current_stressor:
                # If in the advice support, check whether the advice is accepted by user
                if "advice" in self.current_support:
                    if not self.advice_loop:
                        # Accept
                        if "好" in sentence or "不錯" in sentence:
                            response = "好的，希望這個建議能順利幫到你。"
                            self.current_support = list()
                            # self.current_stressor = None
                            print(
                                "Response in Emotional Support System (Advice Accept): ",
                                response,
                            )
                        # Check whether user reject the advice
                        elif "不" in sentence:
                            response = "你是指不要這個建議嗎？你可以說是的或是不是。"
                            # self.temp_sentence = self.current_stressor + '讓' + sentence
                            self.temp_sentence = sentence
                            self.advice_loop = True
                            print(
                                "Response in Emotional Support System (Reject Check): ",
                                response,
                            )
                        else:
                            # # sentence = self.current_stressor + '讓' + sentence
                            # general_response = list(
                            #     self.NLU.splitSentence(self.GC.respond(sentence))
                            # )
                            # # general_response = ''.join(general_response[0]) if len(general_response) > 3 else ''.join(general_response[0:3])
                            # general_response = "".join(general_response[0:])

                            general_response = self.GC.respond(sentence)

                            (
                                es_response,
                                current_stressor,
                                current_support,
                                special_mark,
                            ) = self.ESC.respond(
                                sentence,
                                None,
                                None,
                                ["empathy", "advice", "encouragement"],
                                current_id,
                                self.current_stressor,
                            )

                            if special_mark:
                                self.current_stressor = current_stressor
                                self.current_support = current_support
                                response = (
                                    es_response if es_response else general_response
                                )
                            else:
                                self.current_stressor = None
                                self.current_support = list()
                                response = general_response
                            # response = es_response if es_response else general_response
                            # self.GC.respond(response)
                            print(
                                "Response in Emotional Support System (Other in Advice): ",
                                response,
                                ";",
                                current_stressor,
                            )

                    else:
                        # Change the advice since user reject the advice
                        if "是的" in sentence:
                            input_sentence = (
                                "能提供我解決" + self.current_stressor + "這件事的意見嗎"
                            )

                            # general_response = list(
                            #     self.NLU.splitSentence(self.GC.respond(input_sentence))
                            # )
                            # # general_response = ''.join(general_response[0]) if len(general_response) > 3 else ''.join(general_response[0:3])
                            # general_response = "".join(general_response[0:])
                            general_response = self.GC.respond(sentence)

                            (
                                es_response,
                                current_stressor,
                                current_support,
                                special_mark,
                            ) = self.ESC.respond(
                                input_sentence,
                                None,
                                None,
                                ["advice"],
                                current_id,
                                self.current_stressor,
                            )

                            if es_response:
                                response = "好唷！那" + es_response
                            else:
                                # print('123')
                                (
                                    es_response,
                                    current_stressor,
                                    self.current_support,
                                    special_mark,
                                ) = self.ESC.respond(
                                    input_sentence,
                                    None,
                                    None,
                                    ["encouragement"],
                                    current_id,
                                    self.current_stressor,
                                )
                                if es_response:
                                    response = es_response
                                else:
                                    # self.current_stressor = None
                                    self.current_support = list()
                                    response = general_response

                            print(
                                "Response in Emotional Support System (Another Advice): ",
                                response,
                                ";",
                                current_stressor,
                            )
                            self.advice_loop = False
                        # User does not reject advice; Keep chatting
                        elif "不是" in sentence:
                            # general_response = list(
                            #     self.NLU.splitSentence(
                            #         self.GC.respond(self.temp_sentence)
                            #     )
                            # )
                            # # general_response = ''.join(general_response[0]) if len(general_response) > 3 else ''.join(general_response[0:3])
                            # general_response = "".join(general_response[0:])
                            general_response = self.GC.respond(sentence)

                            (
                                es_response,
                                current_stressor,
                                current_support,
                                special_mark,
                            ) = self.ESC.respond(
                                self.temp_sentence,
                                None,
                                None,
                                ["empathy", "advice", "encouragement"],
                                current_id,
                                self.current_stressor,
                            )

                            if special_mark:
                                self.current_stressor = current_stressor
                                self.current_support = current_support
                                response = (
                                    es_response if es_response else general_response
                                )
                            else:
                                # self.current_stressor = None
                                self.current_support = list()
                                response = general_response
                            # response = es_response if es_response else general_response
                            print(
                                "Response in Emotional Support System (Other in Advice loop): ",
                                response,
                                ";",
                                current_stressor,
                            )
                            self.advice_loop = False
                        else:
                            response = "是還是不是？你可以說是的或是不是。"
                            print("Response in Emotional Support System: ", response)

                else:
                    # # sentence = self.current_stressor + '讓' + sentence
                    # general_response = list(
                    #     self.NLU.splitSentence(self.GC.respond(sentence))
                    # )
                    # # general_response = ''.join(general_response[0]) if len(general_response) > 3 else ''.join(general_response[0:3])
                    # general_response = "".join(general_response[0:])
                    general_response = self.GC.respond(sentence)

                    (
                        es_response,
                        current_stressor,
                        current_support,
                        special_mark,
                    ) = self.ESC.respond(
                        sentence,
                        None,
                        None,
                        ["empathy", "advice", "encouragement"],
                        current_id,
                        self.current_stressor,
                    )

                    if special_mark:
                        self.current_stressor = current_stressor
                        self.current_support = current_support
                        response = es_response if es_response else general_response
                    else:
                        self.current_stressor = None
                        self.current_support = list()
                        response = general_response
                    # response = es_response if es_response else general_response
                    # self.GC.respond(response)
                    print(
                        "Response in Emotional Support System (Other): ",
                        response,
                        ";",
                        current_stressor,
                    )
                    # if not 'advice' in self.current_support:
            else:
                # response = list(self.NLU.splitSentence(self.GC.respond(sentence)))
                # # response = ''.join(response[0]) if len(response) > 3 else ''.join(response[0:3])
                # response = "".join(response[0:])
                response = self.GC.respond(sentence)

                print(
                    "Response in Emotional Support System (Without Stressor): ",
                    response,
                )
                # self.current_stressor = None
                self.current_support = list()

            if "拜拜" in sentence or "再见" in sentence:
                self.current_stressor = None
                self.current_support = list()
                response = "再見，希望你能順利度過困境，下次難過的時候，隨時歡迎你繼續找我聊天"

        print("==============================================")

        # else:
        # 	response = list(self.NLU.splitSentence(self.GC.respond(sentence)))
        # 	response = ''.join(response[0]) if len(response) > 3 else ''.join(response[0:3])
        # 	print('Postive Response in Emotional Support System: ', response)
        return response
