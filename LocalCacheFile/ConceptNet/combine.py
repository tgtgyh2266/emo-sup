import pickle
import os

conceptNet = dict()

file_list = [file for file in os.listdir("./") if file.endswith(".pickle")]

for index, file in enumerate(file_list):
    sub_net = pickle.load(open(file, "rb"))
    if index == 0:
        conceptNet = sub_net
    else:
        conceptNet = {**conceptNet, **sub_net}

with open("conceptNetCache.pickle", "wb") as file:
    pickle.dump(conceptNet, file)
