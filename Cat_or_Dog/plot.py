import os
import pickle

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
    return pickle_dict


def plot_dataSize(data):
    ys = {}
    for k in data:
        if k.startswith("cat_dog_datasize"):
            split_k = k.split("_")
            dataSize = int(split_k[3])
            if "car_bike" in k:
                nEpochs = int(split_k[7])
            else:
                nEpochs = 0
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
    plt.xlabel("Time")
    plt.ylabel("Accuracy")
    plt.legend()
    tikzplotlib.save("exp_time.tex")
    plt.show()


if __name__ == '__main__':
    # data = parse_directory(sys.argv[1])
    data = parse_directory("/home/maxime/Téléchargements/model (1)/")
    #plot_dataSize(data)
    plot_time(data)
