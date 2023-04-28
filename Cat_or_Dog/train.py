import argparse
import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, layers, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
)

SAVE_DIR = "backup"  # Save directory for backup weights during the training


class DogCatClassifier:
    """
    Image classifier for dog and cat pictures using Deep Learning
    Convolutionnal Neural Network
    """

    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    BATCH_SIZE = 32

    def __init__(self, data_dir, data_size=-1, categories=["cat", "dog"], model=None, tl=False, numLayersNotFreezed=1):
        """
        :param data_dir: directory of the data
        """
        self.data_dir = data_dir
        files_path = os.listdir(self.data_dir)
        random.shuffle(files_path)
        self.data_files = files_path[:data_size]

        # Load data and labels
        self.X = self.data_files  # Files names of the images
        random.shuffle(self.X)
        self.X = self.X
        self.y = np.empty(len(self.X), dtype=str)  # Labels

        for category in categories:
            self.y[np.char.startswith(self.X, category)] = category[0]

        self.total_epochs = 0

        self.model = self._load_model(model, tl, numLayersNotFreezed)

        self.history = {
            "n_epochs": 0,
            "accuracy": [],
            "val_accuracy": [],
            "loss": [],
            "val_loss": []
        }

    def fit(self, folder, epochs=1, plot_res_path=os.path.join(SAVE_DIR, "results.png")):
        """Fit the model using the data in the selected directory"""
        train_set, val_set, test_set = self._gen_data()

        # callback object to save weights during the training
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(SAVE_DIR, "weights-{epoch:03d}.ckpt"),
            save_weights_only=True,
            verbose=1,
        )

        # Fit the model
        history = self.model.fit(
            train_set,
            epochs=epochs,
            validation_data=val_set,
            callbacks=[cp_callback],
        )
        self.total_epochs += epochs

        # Show the predictions on the testing set
        result = self.model.evaluate(test_set, batch_size=self.BATCH_SIZE)
        print(
            "Testing set evaluation:",
            dict(zip(self.model.metrics_names, result)),
        )

        # Save model information
        self.model.save(folder)

        self.history["n_epochs"] += epochs
        for key in self.history:
            if key=="n_epochs":
                continue
            self.history[key] += history.history[key]

        # Plot training results
        epochs_range = range(1, self.history["n_epochs"]+1)

        # Accuracy in training and validation sets as the training goes
        acc = self.history["accuracy"]
        val_acc = self.history["val_accuracy"]
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label="Training Accuracy")
        plt.plot(epochs_range, val_acc, label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.title("Training and Validation Accuracy")

        # Loss in training and validation sets as the training goes
        loss = self.history["loss"]
        val_loss = self.history["val_loss"]
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label="Training Loss")
        plt.plot(epochs_range, val_loss, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.title("Training and Validation Loss")

        plt.savefig(plot_res_path)

        directory = os.path.dirname(plot_res_path)
        with open(os.path.join(directory, "data.bin"), "wb") as f:
            pickle.dump(self.history, f)

    def _load_model(self, path, transferlearning, numLayersNotFreezed):
        """Build a CNN model for image classification"""
        if path is None:
            model = Sequential()

            # 2D Convolutional layer
            model.add(
                Conv2D(
                    128,  # Number of filters
                    (3, 3),  # Padding size
                    input_shape=(
                        self.IMG_HEIGHT,
                        self.IMG_WIDTH,
                        3,
                    ),  # Shape of the input images
                    activation="relu",  # Output function of the neurons
                    padding="same",
                )
            )  # Behaviour of the padding region near the borders
            # 2D Pooling layer to reduce image shape
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            # Transform 2D input shape into 1D shape
            model.add(Flatten())
            # Dense layer of fully connected neurons
            model.add(Dense(128, activation="relu"))
            # Dropout layer to reduce overfitting, the argument is the proportion of random neurons ignored in the training
            model.add(Dropout(0.2))
            # Output layer
            model.add(Dense(1, activation="sigmoid"))
            model.compile(
                loss="binary_crossentropy",  # Loss function for binary classification
                optimizer=RMSprop(
                    learning_rate=1e-3
                ),  # Optimizer function to update weights during the training
                metrics=["accuracy", "AUC"],
            )  # Metrics to monitor during training and testing
        else:
            model = tf.keras.models.load_model(path)

        if transferlearning:
            for layer in model.layers[1:len(model.layers) - numLayersNotFreezed]:
                layer.trainable = False

        # Print model summary
        model.summary()
        return model

    def _gen_data(self):
        """Split the data set into training, validation and testing sets"""

        # Split data into training+validation and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y)
        df_train = pd.DataFrame({"filename": X_train, "class": y_train})
        df_test = pd.DataFrame({"filename": X_test, "class": y_test})

        # use data generators as input for the model
        train_datagen = ImageDataGenerator(
            rescale=1 / 255,  # Divide input values by 255 so it ranges between 0 and 1
            # The images are converted from RGB to BGR, then each color channel is
            # zero-centered with respect to the ImageNet dataset, without scaling.
            preprocessing_function=preprocess_input,
            validation_split=0.2,  # Size of the validation set
            horizontal_flip=True,  # Includes random horizontal flips in the data set
            shear_range=0.2,  # Includes random shears in the data set
            height_shift_range=0.2,  # Includes random vertical shifts in the data set
            width_shift_range=0.2,  # Includes random horizontal shifts in the data set
            zoom_range=0.2,  # Includes random zooms in the data set
            rotation_range=30,  # Includes random rotations in the data set
            # Filling methods for undefined regions upon data augmentation
            fill_mode="nearest",
        )
        test_datagen = ImageDataGenerator(
            rescale=1 / 255, preprocessing_function=preprocess_input
        )

        # Load images in the data generators
        train_data_generator = train_datagen.flow_from_dataframe(
            df_train,
            # Directory in which the files can be found
            directory=self.data_dir,
            # Column name for the image names
            x_col="filename",
            # Column name for the labels
            y_col="class",
            # Type of subset
            subset="training",
            # Shuffle the data to avoid fitting the image order
            shuffle=True,
            # batch size
            batch_size=self.BATCH_SIZE,
            # Classification mode
            class_mode="binary",
            # Target size of the images
            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
        )
        valid_data_generator = train_datagen.flow_from_dataframe(
            df_train,
            directory=self.data_dir,
            x_col="filename",
            y_col="class",
            subset="validation",
            shuffle=True,
            batch_size=self.BATCH_SIZE,
            class_mode="binary",
            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
        )
        test_data_generator = test_datagen.flow_from_dataframe(
            df_test,
            directory=self.data_dir,
            x_col="filename",
            y_col="class",
            shuffle=False,
            batch_size=self.BATCH_SIZE,
            class_mode="binary",
            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
        )

        return train_data_generator, valid_data_generator, test_data_generator


class DogCatClassifierKerasArch(DogCatClassifier):
    def __init__(self, data, architecture, data_size=-1, categories=["cat", "dog"], tl=True):
        self.buildArchitecture = architecture
        super().__init__(data, data_size=data_size, categories=categories, tl=tl)

    def _load_model(self, _, transferlearning, ___):
        """Build a CNN model for image classification"""
        # From guide : https://keras.io/guides/transfer_learning/

        w = "imagenet" if transferlearning else None
        # First, instantiate a base model with pre-trained weights.
        base_model = self.buildArchitecture(weights=w,
                                       include_top=False,
                                       input_shape=(DogCatClassifier.IMG_WIDTH, DogCatClassifier.IMG_HEIGHT, 3),
                                       classes=2)
        # Then, freeze the base model.
        base_model.trainable = not transferlearning

        inputs = Input(shape=(DogCatClassifier.IMG_WIDTH, DogCatClassifier.IMG_HEIGHT, 3))
        # We make sure that the base_model is running in inference mode here,
        # by passing `training=False`. This is important for fine-tuning, as you will
        # learn in a few paragraphs.
        x = base_model(inputs, training=not transferlearning)
        # Convert features of shape `base_model.output_shape[1:]` to vectors
        x = layers.GlobalAveragePooling2D()(x)

        # A Dense classifier with a single unit (binary classification)
        outputs = layers.Dense(1)(x)
        model = Model(inputs, outputs)

        model.compile(
            loss="binary_crossentropy",  # Loss function for binary classification
            optimizer=RMSprop(
                learning_rate=1e-3
            ),  # Optimizer function to update weights during the training
            metrics=["accuracy", "AUC"],
        )  # Metrics to monitor during training and testing

        # Print model summary
        model.summary()

        return model


if __name__ == "__main__":
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

    print("########### Exp 1: train model on car and bikes ###########")
    clf = DogCatClassifier(args.data, categories=["Car", "Bike"])
    prev_epoch = 0
    for epoch in [5, 10, 15]:
        model_name = f"car_bike_epoch_{epoch}"
        save_dir = f"{args.folder}/{model_name}"
        plot_path = f"{save_dir}/{model_name}.png"
        time_path = f"{save_dir}/{model_name}.txt"
        os.makedirs(save_dir, exist_ok=True)
        
        start = time.time()
        clf.fit(save_dir, epochs=epoch - prev_epoch, plot_res_path=plot_path)
        prev_epoch = epoch
        end = time.time()
        with open(time_path, 'w') as f:
            f.write(str(end - start) + "\n")

    print("########### Exp 2: transfer learning from car and bikes on cat an dogs ###########")
    for data_size in [100, 500, 1000, 5000, 10000]:
        for motor_bike_epoch in [5, 10, 15]:
            model_name_saved = f"car_bike_epoch_{motor_bike_epoch}"
            model_saved_path = f"{args.folder}/{model_name_saved}"

            model_name = f"cat_dog_datasize_{data_size}_car_bike_epoch_{motor_bike_epoch}"
            save_dir = f"{args.folder}/{model_name}"
            plot_path = f"{save_dir}/{model_name}.png"
            time_path = f"{save_dir}/{model_name}.txt"
            os.makedirs(save_dir, exist_ok=True)
            
            start = time.time()
            clf = DogCatClassifier(args.tldata, data_size=data_size, model=model_saved_path, tl=True)
            clf.fit(save_dir, epochs=15, plot_res_path=plot_path)
            end = time.time()
            with open(time_path, 'w') as f:
                f.write(str(end - start) + "\n")

    print("########### Exp 3: train model on car and bikes ###########")
    clf = DogCatClassifier(args.tldata)
    prev_epoch = 0
    for epoch in [15]:
        model_name = f"cat_dog_epoch_{epoch}"
        save_dir = f"{args.folder}/{model_name}"
        plot_path = f"{save_dir}/{model_name}.png"
        time_path = f"{save_dir}/{model_name}.txt"
        os.makedirs(save_dir, exist_ok=True)

        start = time.time()
        clf.fit(save_dir, epochs=epoch - prev_epoch, plot_res_path=plot_path)
        prev_epoch = epoch
        end = time.time()
        with open(time_path, 'w') as f:
            f.write(str(end - start) + "\n")

    print("########### Exp 4: train model from Keras architecture on car and bikes ###########")
    clf = DogCatClassifierKerasArch(args.tldata, MobileNetV2, tl=False)
    prev_epoch = 0
    for epoch in [15]:
        model_name = f"MobileNetV2_cat_dog_epoch_{epoch}"
        save_dir = f"{args.folder}/{model_name}"
        plot_path = f"{save_dir}/{model_name}.png"
        time_path = f"{save_dir}/{model_name}.txt"
        os.makedirs(save_dir, exist_ok=True)

        start = time.time()
        clf.fit(save_dir, epochs=epoch - prev_epoch, plot_res_path=plot_path)
        end = time.time()
        with open(time_path, 'w') as f:
            f.write(str(end - start) + "\n")
        prev_epoch = epoch

    print("########### Exp 5: transfer learning from Keras architecture on cat an dogs ###########")
    for data_size in [100, 500, 1000, 5000, 10000]:
        for motor_bike_epoch in [5, 10, 15]:
            model_name_saved = f"car_bike_epoch_{motor_bike_epoch}"
            model_saved_path = f"{args.folder}/{model_name_saved}"

            model_name = f"MobileNetV2_cat_dog_datasize_{data_size}_car_bike_epoch_{motor_bike_epoch}"
            save_dir = f"{args.folder}/{model_name}"
            plot_path = f"{save_dir}/{model_name}.png"
            time_path = f"{save_dir}/{model_name}.txt"
            os.makedirs(save_dir, exist_ok=True)

            start = time.time()
            clf = DogCatClassifierKerasArch(args.tldata, MobileNetV2, data_size=data_size)
            clf.fit(save_dir, epochs=15, plot_res_path=plot_path)
            end = time.time()
            with open(time_path, 'w') as f:
                f.write(str(end - start) + "\n")
