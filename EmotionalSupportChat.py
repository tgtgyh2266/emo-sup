# -*- coding: UTF-8 -*-

import zhconv
import pickle
from random import *


class EmotionalSupportChat:
    def __init__(self):
        self.stressor = None
        self.feeling = None
        self.emotionalsupport = None
        self.record_support = dict()
        self.record_response = dict()

        with open("./LocalCacheFile/emotionalSupportResponse.pickle", "rb") as file:
            self.es_chat = pickle.load(file)

        with open("./LocalCacheFile/negativeEmotion.pickle", "rb") as file:
            self.feeling_word = pickle.load(file)

    def transform_response(self, response, emotionalsupport):
        if emotionalsupport == "empathy" and self.feeling:
            feeling_origin_senetence = [
                feeling for feeling in self.feeling_word if feeling in response
            ]
            specific_feeling = self.feeling[randint(0, min(4, len(self.feeling) - 1))][
                0
            ]

            for f in feeling_origin_senetence:
                if not f == "寂寞":
                    response = response.replace(f, specific_feeling)

        return response

    def checkrepository(self, observation):
        set_observation = set(observation)
        set_stressor = set(self.es_chat.keys())
        union_set = set_observation & set_stressor

        if union_set:
            return True
        else:
            return False

    def respond(
        self,
        sentence,
        concept_keys,
        feeling,
        emotionalsupport,
        current_id,
        stressor=None,
    ):
        response = None
        # print(self.record_support, self.record_response)
        self.feeling = feeling
        self.emotionalsupport = emotionalsupport
        self.stressor = None
        self.current_id = current_id
        special_mark = False
        sentence = zhconv.convert(sentence, "zh-tw")
        # print("Concept Keywords: ", concept_keys)

        # Select stressor from keywords (The maximum length of keyword and also in the emotional support response)
        # if stressor and (not ("謝" in sentence or "感激" in sentence)) :
        # 	self.stressor = stressor
        while True:
            if not concept_keys:
                break

            key = max(concept_keys, key=len)

            if key in self.es_chat:
                self.stressor = key
                break
            else:
                del concept_keys[key]

        if (
            (not self.stressor)
            and stressor
            and (not ("謝" in sentence or "感激" in sentence))
        ):
            self.stressor = stressor

        print("Current Stressor: ", self.stressor)
        print("==============================================")

        # Record the used emotional support methods
        if not self.current_id in self.record_support:
            self.record_support[self.current_id] = dict()
            self.record_response[self.current_id] = dict()
        if self.stressor and not self.stressor in self.record_support[self.current_id]:
            self.record_support[self.current_id][self.stressor] = list()
            self.record_response[self.current_id][self.stressor] = dict()
            self.record_response[self.current_id][self.stressor]["empathy"] = list()
            self.record_response[self.current_id][self.stressor]["advice"] = list()
            self.record_response[self.current_id][self.stressor][
                "encouragement"
            ] = list()

        # Remove the used emotional support methods
        if self.stressor and self.record_support[self.current_id][self.stressor]:
            self.emotionalsupport = list(
                set(self.emotionalsupport)
                - set(self.record_support[self.current_id][self.stressor])
            )

        # If the user tell his/her feelings, remove 'empathy' method
        empathy_signal = (
            True if sum([word in sentence for word in self.feeling_word]) > 1 else False
        )
        if "empathy" in self.emotionalsupport and empathy_signal:
            self.emotionalsupport.remove("empathy")

        # Randomly select one emotional support methods in candidate set
        if (
            "意見" in sentence
            or "建議" in sentence
            or "方式" in sentence
            or "方法" in sentence
            or "策略" in sentence
        ) and "advice" in emotionalsupport:
            select_es = ["advice"]
            special_mark = True
        elif "鼓勵" in sentence and "encouragement" in emotionalsupport:
            select_es = ["encouragement"]
            special_mark = True
        elif self.emotionalsupport:
            if stressor and not concept_keys:
                return None, None, None, False

            # if 'empathy' in self.emotionalsupport:
            # 	select_es = ['empathy']
            # else:
            # 	if 'encouragement' in self.emotionalsupport:
            # 		select_es = ['encouragement']
            # 	else:
            select_es = [
                self.emotionalsupport[randint(0, len(self.emotionalsupport) - 1)]
            ]

            # # Multiple choice of emotional support: e, e+a, e+en
            # if 'empathy' in select_es:
            # 	if randint(0, 1) and len(self.emotionalsupport) >= 2:
            # 		self.emotionalsupport.remove('empathy')
            # 		select_es.append(self.emotionalsupport[randint(0, len(self.emotionalsupport)-1)])
        else:
            return response, None, None, False

        if self.stressor == stressor and (not special_mark):
            return response, None, None, False

        # print("Select Stressor: {} ; Select_EmotionalSupport: {}".format(self.stressor, select_es))

        # if self.stressor == '爭執':
        # 	select_es = ['advice']

        response_list = list()
        # Select one sentence in the emotional support response and update the used method
        for es in select_es:
            #  Remove the used emotional support sentence
            if self.stressor in self.es_chat and self.es_chat[self.stressor][es]:
                chat_list = list(
                    set(self.es_chat[self.stressor][es])
                    - set(self.record_response[self.current_id][self.stressor][es])
                )

                if chat_list:
                    num_sentence = len(chat_list)
                    select_idx = randint(0, num_sentence - 1)
                    response_list.append(
                        self.transform_response(chat_list[select_idx], es)
                    )
                    self.record_support[self.current_id][self.stressor].append(es)
                    self.record_response[self.current_id][self.stressor][es].append(
                        chat_list[select_idx]
                    )

                    # Because '要不要跟我聊聊' is also a advice ( in the empathy response )
                    if "要不要跟我聊聊" in chat_list[select_idx]:
                        break

        response = "。".join(rp for rp in response_list)

        return response, self.stressor, select_es, special_mark


def main():
    ESC = EmotionalSupportChat()
    # print(ESC.es_chat)
    # s = input('Stressor: ')
    # s = "考不好"
    # f = input('Feeling: ')
    # f = ["生氣"]
    # es = input('EmotionalSupport: ')
    # es = ["empathy"]
    # es = ["advice"]
    # es = ["empathy", "advice", "encouragement"]
    # ESC.respond(s, f, es)
    # print(ESC.respond(s, f, es))

    sentence = input('input sentence: ')
    concept_keys = {sentence:1}
    feeling = None
    emotional_support = ["empathy", "advice", "encouragement"]
    current_id = None
    current_stressor = None
    (
        es_response,
        current_stressor,
        current_support,
        special_mark,
    )= ESC.respond(
        sentence,
        concept_keys,
        feeling,
        emotional_support,
        current_id,
        current_stressor,
    )
    print(es_response)


if __name__ == "__main__":
    main()
