import os
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(11, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))

        return x


class DecisionModule:
    def __init__(self):
        self.emotional_support = ["empathy", "advice", "encouragement"]
        self.model = NeuralNetwork()
        self.model.cuda()
        self.model.load_state_dict(torch.load("./model/nn_Maxformat_distribution.pkl"))

    def output_transform(self, prob_vector):
        # Output to Emotional support categories
        es_dict = dict()
        select_vector = [
            index for index, result in enumerate(prob_vector) if result >= 0.5
        ]

        for index in select_vector:
            es_dict[self.emotional_support[index]] = prob_vector[index]

        output_vector = sorted(es_dict.items(), key=lambda d: d[1], reverse=True)
        output_list = [es_info[0] for es_info in output_vector]

        return output_list

    def model_predict(self, input_vector):
        input_tensor = torch.as_tensor(input_vector, dtype=torch.float)
        input_tensor = input_tensor.cuda()
        predict_tensor = self.model(input_tensor)
        predict_vector = predict_tensor.cpu().data.numpy()[0]
        # select_vector = [index for index, result in enumerate(predict_vector) if result >= 0.5]
        output_vector = self.output_transform(predict_vector)

        return output_vector


def to_categorical(input_class, num_classes):
    res = np.zeros(num_classes, dtype=int)
    res[input_class] = 1
    return res


def main():
    sv = input("Stressor: ")
    pv = input("Personality trait: ")

    sv = to_categorical(int(sv), 6)
    pv = to_categorical(int(pv), 5)
    vector = np.expand_dims(np.concatenate([sv, pv], axis=-1), axis=0)

    DM = DecisionModule()
    o = DM.model_predict(vector)
    print(o)


if __name__ == "__main__":
    main()
