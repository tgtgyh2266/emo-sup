# -*- coding: UTF-8 -*-

import zhconv
import pickle
import os
import codecs

from pyltp import Segmentor
from pyltp import SentenceSplitter
from pyltp import Postagger
from pyltp import Parser
from pyltp import NamedEntityRecognizer
from pyltp import SementicRoleLabeller

model_dir = "./model/"


class NaturalLanguageUnderstanding:
    def __init__(self):
        self.stressors = [
            "Physiological",
            "Frustration",
            "Pressure",
            "Conflict",
            "Change",
            "Ioslation",
        ]
        self.stressor_graph = pickle.load(
            open("./LocalCacheFile/initialConceptGraph_Stressor.pickle", "rb")
        )
        self.stressor_seeds = pickle.load(
            open("./LocalCacheFile/showSeeds.pickle", "rb")
        )
        self.stop_words = self.loadTxt("./model/stopwords.txt")
        self.keywords = dict()
        self.nPOS = ["c", "e", "g", "h", "m", "nh", "nt", "q"]

        with open("./LocalCacheFile/emotionalSupportResponse.pickle", "rb") as file:
            self.es_chat = pickle.load(file)

    def loadTxt(self, path):
        stop_words = set()
        for word in codecs.open(path, "r", "utf-8", "ignore"):
            stop_words.add(word.strip())
        return stop_words

    def splitSentence(self, text):
        sentences = SentenceSplitter.split(text)
        return sentences

    def segmentSentence(self, sentence):
        segmentor = Segmentor()
        segmentor.load(os.path.join(model_dir, "cws.model"))
        words = segmentor.segment(sentence)
        segmentor.release()
        return words

    def postagWord(self, words):
        postagger = Postagger()
        postagger.load(os.path.join(model_dir, "pos.model"))
        postags = postagger.postag(words)
        postagger.release()
        return postags

    def parserWord(self, words, postags):
        parser = Parser()
        parser.load(os.path.join(model_dir, "parser.model"))
        arcs = parser.parse(words, postags)
        parser.release()
        return arcs

    def sementicWord(self, words, postags, arcs):
        labeller = SementicRoleLabeller()
        labeller.load(os.path.join(model_dir, "pisrl.model"))
        roles = labeller.label(words, postags, arcs)
        labeller.release()
        return roles

    def capture_key_words(self, text):
        self.keywords = dict()

        words = self.segmentSentence(text)
        postags = self.postagWord(words)
        arcs = self.parserWord(words, postags)
        self.keywords = self.processWord(words, postags, arcs)
        print("Original Keywords: ", self.keywords)
        self.keywords = self.capture_matchWord(text)

        # 2021/06/18 add : check if sentence contains keywords of template response
        es_keywords = self.es_chat.keys()
        for key in es_keywords:
            text = zhconv.convert(text, "zh-tw")
            if key in text:
                self.keywords[key] = 1

        return self.keywords

    def processWord(self, words, postags, arcs):
        words = list(words)
        postags = list(postags)
        arcs = [(arc.head, arc.relation) for arc in arcs]

        # Remove the stopwords
        index = 0
        while index < len(words):
            if words[index] in self.stop_words or postags[index] in self.nPOS:
                words.pop(index)
                postags.pop(index)
                arcs.pop(index)
                index -= 1
            index += 1

        index = 0
        temp_word = None
        keywords = dict()
        while index < len(arcs):
            if arcs[index][1] in ["HED", "COO"]:
                root_word = words[index]

                # Combine words
                if index > 0:
                    if arcs[index - 1][1] in ["VOB", "ADV", "CMP"]:
                        root_word = words[index - 1] + root_word
                        words.pop(index - 1)
                        postags.pop(index - 1)
                        arcs.pop(index - 1)
                        index -= 1

                if index < len(words) - 1:
                    if arcs[index + 1][1] in ["VOB", "ADV", "CMP"]:
                        root_word = root_word + words[index + 1]
                        words.pop(index + 1)
                        postags.pop(index + 1)
                        arcs.pop(index + 1)
                keywords[root_word] = 1
                words.pop(index)
                postags.pop(index)
                arcs.pop(index)
                index -= 1
            index += 1

        # Remain Item
        sum_weight = 0
        for index, word in enumerate(words):
            if postags[index] != "wp":
                keywords[word] = 1

        return keywords

    def capture_matchWord(self, text):
        # Capture word in stressor graph; to Traditional Chinese
        matchwords = dict()
        spare_matchwords = dict()
        for word in self.stressor_graph.nodes():
            word = zhconv.convert(word, "zh-cn")
            if word in text:
                tw_word = zhconv.convert(word, "zh-tw")
                spare_matchwords[tw_word] = 1
                if not sum([1 for keyword in self.keywords if word in keyword]):
                    tw_word = zhconv.convert(word, "zh-tw")
                    matchwords[tw_word] = 1
            if word in self.keywords:
                tw_word = zhconv.convert(word, "zh-tw")
                matchwords[tw_word] = 1
        if len(matchwords) == 0:
            matchwords = spare_matchwords

        return matchwords


def main():
    NLU = NaturalLanguageUnderstanding()
    text = input("Input text: ")
    text = zhconv.convert(text, "zh-cn")
    sentences = NLU.splitSentence(text)
    # print(list(sentences))
    print(NLU.capture_key_words(text))
    keyword_list = []

    # for sentence in sentences:
    # 	words = NLU.segmentSentence(sentence)
    # 	postags = NLU.postagWord(words)
    # 	print(list(words), list(postags))
    # 	arcs = NLU.parserWord(words, postags)
    # 	print([(arc.head, arc.relation) for arc in arcs])


if __name__ == "__main__":
    main()
