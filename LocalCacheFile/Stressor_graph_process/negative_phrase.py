# -*- coding: UTF-8 -*-

import pickle

with open("./phraseTypeCache.pickle", "rb") as file:
    table_phrase_type = pickle.load(file)

for word in table_phrase_type:
    if table_phrase_type[word] == "emotion":
        print("123")
