import time
from keras.applications import MobileNetV2

from train import *

parser = argparse.ArgumentParser(description="CNN Trainer for the Cat or Dog app.")

parser.add_argument(
    "-f",
    "--folder",
    type=str,
    help="Destination folder to save the model after training ends",
    default="Custom",
)

parser.add_argument(
    "-d",
    "--data",
    type=str,
    help="Images folder for the base model",
    required=True,
)

parser.add_argument(
    "-tld",
    "--tldata",
    type=str,
    help="Images folder for the transfer learning model",
    required=True,
)

args = parser.parse_args()

val_set_size = 200
test_set_size = 100

print("########### Exp 1: train book models on car and bikes ###########")
clf = DogCatClassifier(args.data, test_set_size=test_set_size, val_set_size=val_set_size, categories=["Car", "Bike"])
prev_epoch = 0
for epoch in [5, 10, 15]:
    model_name = f"car_bike_epoch_{epoch}"
    save_dir = f"{args.folder}/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    clf.fit(save_dir, epochs=epoch - prev_epoch)
    prev_epoch = epoch
    end = time.time()

print("########### Exp 2: transfer learning from car and bikes book models on cat an dogs ###########")
for data_size in [100, 500, 1000, 5000, 10000-(test_set_size+val_set_size)]:
    for motor_bike_epoch in [5, 10, 15]:
        model_name_saved = f"car_bike_epoch_{motor_bike_epoch}"
        model_saved_path = f"{args.folder}/{model_name_saved}"

        model_name = f"cat_dog_datasize_{data_size}_car_bike_epoch_{motor_bike_epoch}"
        save_dir = f"{args.folder}/{model_name}"
        os.makedirs(save_dir, exist_ok=True)

        clf = DogCatClassifier(args.tldata, test_set_size=test_set_size, val_set_size=val_set_size, train_set_size=data_size, model=model_saved_path, tl=True)
        clf.fit(save_dir, epochs=15)


print("########### Exp 3: train book models on cat and dog ###########")
for data_size in [100, 500, 1000, 5000, 10000-(test_set_size+val_set_size)]:
    clf = DogCatClassifier(args.tldata, test_set_size=test_set_size, val_set_size=val_set_size, train_set_size=data_size)
    prev_epoch = 0
    for epoch in [15]:
        model_name = f"cat_dog_datasize_{data_size}_epoch_{epoch}"
        save_dir = f"{args.folder}/{model_name}"
        os.makedirs(save_dir, exist_ok=True)

        clf.fit(save_dir, epochs=epoch - prev_epoch)
        prev_epoch = epoch

print("########### Exp 4: transfer learning from ImageNet MobileNetV2 on cat an dogs ###########")
for data_size in [100, 500, 1000, 5000, 10000-(test_set_size+val_set_size)]:
    model_name = f"MobileNetV2_cat_dog_datasize_{data_size}_imagenet"
    save_dir = f"{args.folder}/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    clf = DogCatClassifierKerasArch(args.tldata, MobileNetV2, train_set_size=data_size, test_set_size=test_set_size, val_set_size=val_set_size)
    clf.fit(save_dir, epochs=15)
