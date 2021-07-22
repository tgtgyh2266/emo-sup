import os
import pickle
import torch
import sys

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from sklearn.svm import SVC


class NeuralNetwork(nn.Module):
    def __init__(self, format_output):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(11, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)

        if format_output == "class":
            self.fc4 = nn.Linear(1024, 7)
        elif format_output == "distribution":
            self.fc4 = nn.Linear(1024, 3)

        self.format_output = format_output

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        if self.format_output == "class":
            x = self.fc4(x)
            return x
        elif self.format_output == "distribution":
            # x = F.sigmoid(self.fc4(x))
            x = torch.sigmoid(self.fc4(x))
            return x


class FeatureData(Dataset):
    def __init__(self, input_vector, output_vector, format_output):
        self.input_vector = input_vector
        self.output_vector = output_vector
        self.format_output = format_output

    def __len__(self):
        return self.input_vector.shape[0]

    def __getitem__(self, idx):
        self.input_vector = torch.as_tensor(self.input_vector, dtype=torch.float)
        if self.format_output == "class":
            self.output_vector = torch.as_tensor(self.output_vector, dtype=torch.long)
        elif self.format_output == "distribution":
            self.output_vector = torch.as_tensor(self.output_vector, dtype=torch.float)
        return self.input_vector[idx], self.output_vector[idx]


class TrainingDecisionModule:
    def __init__(self):
        self.train_path = "./EmotionalSupport_Questionnaire/Data/Train_new"
        self.validate_path = "./EmotionalSupport_Questionnaire/Data/All_data"
        self.stressor_feature_vector_path = (
            "./EmotionalSupport_Questionnaire/stressor_vector.txt"
        )
        self.trait_weight_path = "./EmotionalSupport_Questionnaire/trait_weight.txt"
        self.model_path = "./EmotionalSupport_Questionnaire/model"
        self.stressor_feature_vector = list()
        self.trait_weight_vector = list()
        self.result_vector = list()
        self.groundtruth_vector = list()
        self.get_stressorvector()

    def to_categorical(self, input_class, num_classes):
        res = np.zeros(num_classes, dtype=int)
        res[input_class] = 1
        return res

    def get_stressorvector(self):
        fp = open(self.stressor_feature_vector_path, "r")
        stressor_content = fp.readlines()
        for stressor_vector in stressor_content:
            temp_content = stressor_vector.split("\n")[0]
            temp_value = temp_content.split(",")
            if len(temp_value) == 6:
                temp_vector = [float(value) for value in temp_value]
                self.stressor_feature_vector.append(temp_vector)
        fp.close()
        # self.stressor_feature_vector = np.loadtxt(self.stressor_feature_vector_path)

    def get_supportvector_classes(self, data_path):
        emotional_support_vector = list()
        for index, path in enumerate(os.listdir(data_path)):
            # print(path)
            emotional_support_vector.append(list())
            file_path = os.path.join(data_path, path)
            fp = open(file_path, "r")
            information = fp.readlines()[3:38]
            for info in information:
                temp_content = info.split("\n")[0]
                temp_value = temp_content.split(",")
                temp_class = 0
                for value in temp_value:
                    if value != "":
                        temp_class += 2 ** int(value)
                emotional_support_vector[index].append(temp_class)
            fp.close()
        return emotional_support_vector

    def get_supportvector_distribution(self, data_path):
        emotional_support_vector = list()
        for index, path in enumerate(os.listdir(data_path)):
            emotional_support_vector.append(list())
            file_path = os.path.join(data_path, path)
            fp = open(file_path, "r")
            information = fp.readlines()[3:38]
            for info in information:
                temp_content = info.split("\n")[0]
                temp_value = temp_content.split(",")
                temp_vector = [0, 0, 0]
                for value in temp_value:
                    if value != "":
                        temp_vector[int(value)] = 1
                emotional_support_vector[index].append(temp_vector)
            fp.close()
        return emotional_support_vector

    def get_traitvector(self, data_path, format_style):
        # Trait_weight
        fp = open(self.trait_weight_path, "r")
        information = fp.readlines()
        for info in information:
            temp_value = info.split("\n")[0]
            self.trait_weight_vector.append(int(temp_value))
        fp.close()
        # print('WeightVector Shape: ', np.array(self.trait_weight_vector).shape)

        # Compute trait_vector
        personality_trait_vector = list()
        for index, path in enumerate(os.listdir(data_path)):
            file_path = os.path.join(data_path, path)
            fp = open(file_path, "r")
            information = fp.readlines()[38:78]
            temp_vector = [16, 4, 22, 16, 16]
            for num, info in enumerate(information):
                temp_value = info.split("\n")[0]
                if num < 8:
                    temp_vector[0] += self.trait_weight_vector[num] * int(temp_value)
                elif 8 <= num < 16:
                    temp_vector[1] += self.trait_weight_vector[num] * int(temp_value)
                elif 16 <= num < 24:
                    temp_vector[2] += self.trait_weight_vector[num] * int(temp_value)
                elif 24 <= num < 32:
                    temp_vector[3] += self.trait_weight_vector[num] * int(temp_value)
                elif 32 <= num < 40:
                    temp_vector[4] += self.trait_weight_vector[num] * int(temp_value)

            if format_style == "Transform":
                transform_vector = np.zeros((5), dtype=np.uint8)
                temp_vector = np.array(temp_vector, dtype=np.uint8)
                transform_vector[temp_vector <= 10] = 0
                transform_vector[temp_vector - 11 <= 10] = 1
                transform_vector[temp_vector - 22 <= 10] = 2
                temp_vector = transform_vector
            elif format_style == "Maxformat":
                temp_vector = np.array(temp_vector, dtype=np.uint8)
                max_index = np.argmax(temp_vector, axis=0)
                temp_vector = self.to_categorical(max_index, 5)
            # print(path, " : ", temp_vector)
            personality_trait_vector.append(temp_vector)
        return personality_trait_vector

    def data_process(self, path, format_style, format_output):
        input_trait = np.array(self.get_traitvector(path, format_style))
        input_trait = np.tile(np.expand_dims(input_trait, axis=1), (1, 35, 1))
        num_individuals = input_trait.shape[0]

        input_stressor = np.array(self.stressor_feature_vector)
        input_stressor = np.tile(
            np.expand_dims(input_stressor, axis=0), (num_individuals, 1, 1)
        )

        # Input: stressor
        if format_style == "None":
            input_feature = input_stressor
        else:
            # Input: stressor + trait
            input_feature = np.concatenate((input_stressor, input_trait), axis=-1)

        input_feature = np.reshape(
            input_feature,
            (input_feature.shape[0] * input_feature.shape[1], input_feature.shape[2]),
        )
        # input_feature = np.expand_dims(input_feature, axis=-1)
        # print('InputVector Shape: ', input_feature.shape)
        if format_output == "class":
            output_support = np.array(self.get_supportvector_classes(path))
            output_support = np.reshape(
                output_support, (output_support.shape[0] * output_support.shape[1])
            )
        elif format_output == "distribution":
            output_support = np.array(self.get_supportvector_distribution(path))
            output_support = np.reshape(
                output_support,
                (
                    output_support.shape[0] * output_support.shape[1],
                    output_support.shape[2],
                ),
            )
        # print('OutputVector Shape: ', output_support.shape)

        return input_feature, output_support

    def svm_model(self, format_style):
        input_feature, output_support = self.data_process(
            self.train_path, format_style, "class"
        )

        clf = SVC(C=1, kernel="rbf", probability=True, gamma=2)
        clf.fit(input_feature, output_support)

        with open(
            os.path.join(self.model_path, "svm_{}_class.pkl".format(format_style)), "wb"
        ) as f:
            pickle.dump(clf, f)

    def nn_model(self, format_style, format_output):
        # Train data
        train_input, train_output = self.data_process(
            self.train_path, format_style, format_output
        )

        print(
            "Example Train Input: {} - Shape: {}".format(
                train_input[0], train_input.shape
            )
        )
        print(
            "Example Train Ouput: {} - Shape: {}".format(
                train_output[0], train_output.shape
            )
        )

        if format_output == "class":
            train_output = train_output - 1

        train_data = FeatureData(train_input, train_output, format_output)
        trainloader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4)

        # Validate data
        validate_input, validate_output = self.data_process(
            self.validate_path, format_style, format_output
        )

        print(
            "Example Validate Input: {} - Shape: {}".format(
                validate_input[0], validate_input.shape
            )
        )
        print(
            "Example Validate Output: {} - Shape: {}".format(
                validate_output[0], validate_output.shape
            )
        )

        validate_data = FeatureData(validate_input, validate_output, format_output)
        validateloader = DataLoader(
            validate_data, batch_size=1, shuffle=False, num_workers=1
        )

        # save validate output
        self.validate_gt(format_output)

        model = NeuralNetwork(format_output)
        model.cuda()

        if format_output == "class":
            criterion = nn.CrossEntropyLoss()
        elif format_output == "distribution":
            criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

        best_loss = np.Infinity
        best_dice = 0.0

        for epoch in range(500):
            for index, data in enumerate(trainloader):
                # get inputs and labels: data = [inputs, labels]
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                if format_output == "class":
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.view(-1,))
                elif format_output == "distribution":
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                # print loss
                if epoch != 0 and index == 0:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                sys.stdout.write(
                    "\rEpoch: {} - Batch: {} => Loss: {}".format(
                        epoch + 1, index + 1, loss.item()
                    )
                )
                sys.stdout.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()

            # Validate
            for index, data in enumerate(validateloader):
                inputs, outputs = data
                inputs = inputs.cuda()

                if format_output == "class":
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.view(-1,))
                elif format_output == "distribution":
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                if format_output == "class":
                    predicts = torch.max(outputs, 1)[1].data.squeeze().item() + 1
                    bin_predict = bin(predicts)[2:]
                    predict_list = [
                        len(bin_predict) - index - 1
                        for index, result in enumerate(bin_predict)
                        if int(result) == 1
                    ]
                    self.result_vector.append(predict_list)
                elif format_output == "distribution":
                    predicts = outputs.cpu().data.numpy()[0]
                    predict_list = [
                        index for index, result in enumerate(predicts) if result >= 0.5
                    ]
                    self.result_vector.append(predict_list)

            # Get evaluation value
            running_dice = self.evaluation_dice()
            if running_dice >= best_dice:
                best_dice = running_dice
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        self.model_path,
                        "nn_{}_{}.pkl".format(format_style, format_output),
                    ),
                )
            sys.stdout.write(
                "Validate -> Epoch: {} => Loss: {}    DICE: {}\n".format(
                    epoch + 1, loss, running_dice
                )
            )
            sys.stdout.flush()
            self.result_vector = list()
        print("Best_Dice: {}".format(best_dice))

    def validate_svm(self, format_style):
        with open(
            os.path.join(self.model_path, "svm_{}_class.pkl".format(format_style)), "rb"
        ) as f:
            clf = pickle.load(f)

        self.validate_gt("class")

        input_feature, output_support = self.data_process(
            self.validate_path, format_style, "class"
        )

        for index in range(len(input_feature)):
            validate_feature = np.expand_dims(input_feature[index], axis=0)

            # Predict
            predict_result = clf.predict(validate_feature)[0]
            bin_result = bin(predict_result)[2:]
            result_list = [
                len(bin_result) - index - 1
                for index, result in enumerate(bin_result)
                if int(result) == 1
            ]
            # print("Result: ", result_list)
            self.result_vector.append(result_list)

        running_dice = self.evaluation_dice()
        print("Dice: {}".format(running_dice))

    def validate_gt(self, format_output):
        if format_output == "class":
            output_support = np.array(
                self.get_supportvector_classes(self.validate_path)
            )
            output_support = np.reshape(
                output_support, (output_support.shape[0] * output_support.shape[1])
            )

            for value in output_support:
                groundtruth_class = value
                bin_groundtruth = bin(groundtruth_class)[2:]
                groundtruth_list = [
                    len(bin_groundtruth) - index - 1
                    for index, result in enumerate(bin_groundtruth)
                    if int(result) == 1
                ]
                self.groundtruth_vector.append(groundtruth_list)
        elif format_output == "distribution":
            output_support = np.array(
                self.get_supportvector_distribution(self.validate_path)
            )
            output_support = np.reshape(
                output_support,
                (
                    output_support.shape[0] * output_support.shape[1],
                    output_support.shape[2],
                ),
            )

            for distribtuion in output_support:
                groundtruth_list = [
                    index for index, result in enumerate(distribtuion) if result != 0
                ]
                self.groundtruth_vector.append(groundtruth_list)

    def evaluation_dice(self):
        dice_value = 0
        for index in range(len(self.groundtruth_vector)):
            # Dice
            dice_denominator = len(self.result_vector[index]) + len(
                self.groundtruth_vector[index]
            )
            intersection = list(
                set(self.result_vector[index]) & set(self.groundtruth_vector[index])
            )
            intersection_value = len(intersection)
            dice_value += 2 * intersection_value / dice_denominator

            # IOU
            # intersection = list(set(self.result_vector[index]) & set(self.groundtruth_vector[index]))
            # intersection_value = len(intersection)
            # union = list(set(self.result_vector[index]) | set(self.groundtruth_vector[index]))
            # union_value = len(union)
            # dice_value += (intersection_value/union_value)

        dice_value = dice_value / len(self.groundtruth_vector)
        return dice_value


def main():
    TDM = TrainingDecisionModule()
    # TDM.get_stressorvector()
    # TDM.svm_model('Maxformat')
    # TDM.validate_svm('Maxformat')
    TDM.nn_model("Maxformat", "distribution")
    # TDM.validate_algorithm()


if __name__ == "__main__":
    main()
