from matplotlib import pyplot as plt
import os
import pickle

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
                ys[nEpochs].append((dataSize, data[k]["val_accuracy"][-1]))
            else:
                ys[nEpochs] = [(dataSize, data[k]["val_accuracy"][-1])]


    for key in ys:
        ys[key].sort(key=lambda tup:tup[0])
        x,y = zip(*ys[key])
        plt.plot(x, y, label="nEpochs base model = {}".format(key), linestyle="-")
    plt.title("TL accuracy in terms of dataset size")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data = parse_directory("/home/emma/Documents/MA2/techniques_ia/model")
    plot_dataSize(data)