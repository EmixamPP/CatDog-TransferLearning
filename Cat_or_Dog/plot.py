import os
import sys
import pickle

import numpy as np
import tikzplotlib
from matplotlib import pyplot as plt


def parse_directory(directory):
    pickle_dict = {}
    files = os.listdir(directory)
    for filename in files:
        file = os.path.join(directory, filename, "data.bin")
        with (open(file, "rb")) as openfile:
            pickle_data = pickle.load(openfile)
            pickle_dict[filename] = pickle_data

        if "time" not in pickle_data:
            with open(os.path.join(directory, filename, filename + ".txt"), "r") as file:
                t = float(file.readline().strip())
                pickle_data["time"] = t

    return pickle_dict


def plot_dataSize(data):
    ys = {}
    for k in data:
        #if k.startswith("cat_dog_datasize"):
        if k.startswith("MobileNetV2_cat_dog_datasize_"):
            split_k = k.split("_")
            dataSize = int(split_k[4])
            """if "car_bike" in k:
                nEpochs = int(split_k[8])
            else:
                nEpochs = 0"""
            nEpochs = 15
            if nEpochs in ys:
                ys[nEpochs].append((dataSize, max(data[k]["val_accuracy"])))
            else:
                ys[nEpochs] = [(dataSize, max(data[k]["val_accuracy"]))]

    for key in ys:
        ys[key].sort(key=lambda tup: tup[0])
        x, y = zip(*ys[key])
        plt.plot(x, y, label="nEpochs base model = {}".format(key), linestyle="-")
    plt.title("TL accuracy in terms of dataset size")
    plt.xlabel("Dataset size")
    plt.ylabel("Accuracy")
    plt.legend()
    tikzplotlib.save("exp_MNet_acc.tex")
    plt.show()


def plot_time(data):
    ys = {}
    for k in data:
        if k.startswith("cat_dog_datasize") or k.startswith("MobileNetV2_cat_dog_datasize_"):
            split_k = k.split("_")
            dataSize = int(split_k[3 + k.startswith("MobileNetV2_cat_dog_datasize_")])
            if "car_bike" in k:
                nEpochs = int(split_k[7 + k.startswith("MobileNetV2_cat_dog_datasize_")])
            else:
                nEpochs = 0

            if k.startswith("MobileNetV2_cat_dog_datasize_"):
                nEpochs = "MobileNetV2"

            if nEpochs in ys:
                ys[nEpochs].append((dataSize, data[k]["time"]))
            else:
                ys[nEpochs] = [(dataSize, data[k]["time"])]

    for key in ys:
        ys[key].sort(key=lambda tup: tup[0])
        x, y = zip(*ys[key])
        plt.plot(x, y, label="nEpochs base model = {}".format(key), linestyle="-")
    plt.title("Training time in terms of dataset size")
    plt.xlabel("Dataset size")
    plt.ylabel("Time (s)")
    plt.legend()
    tikzplotlib.save("exp_time.tex")
    plt.show()

def plot_acc(data, trainingSetSize=100):
    ys = {}
    for k in data:
        if k.startswith("cat_dog_datasize") or k.startswith("MobileNetV2_cat_dog_datasize_"):
            split_k = k.split("_")
            dataSize = int(split_k[3 + k.startswith("MobileNetV2_cat_dog_datasize_")])
            if dataSize != trainingSetSize:
                continue
            if "car_bike" in k:
                nEpochs = int(split_k[7 + k.startswith("MobileNetV2_cat_dog_datasize_")])
            else:
                nEpochs = 0

            if k.startswith("MobileNetV2_cat_dog_datasize_"):
                nEpochs = "MobileNetV2"

            ys[nEpochs] = data[k]["val_accuracy"]

    for key in ys:
        y = [0] + ys[key]
        x = np.arange(len(y))
        plt.plot(x, y, label="nEpochs base model = {}".format(key), linestyle="-")
    plt.title("Accuracy in terms of epochs")
    plt.xlabel("nEpochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.xlim(0, len(x))
    tikzplotlib.save("exp_acc.tex")
    plt.show()

if __name__ == '__main__':
    data = parse_directory(sys.argv[1])
    plot_dataSize(data)
    plot_time(data)
    plot_acc(data, 9700)

