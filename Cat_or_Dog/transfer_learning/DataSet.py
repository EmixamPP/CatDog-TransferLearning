import errno
import glob
import os
import platform

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical


def _get_dir_elem(dataset_path, extension):
    dir_path = dataset_path + "/" + "**/*." + extension
    if platform.system() == "Windows":
        path_symbol = "\\"
        dir_path.replace("/", "\\")
    else:
        path_symbol = "/"
        dir_path.replace("\\", "/")

    dir_elem = glob.glob(dir_path, recursive=True)
    if not len(dir_elem):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dataset_path)

    return dir_elem, path_symbol


class DataSet:
    def __init__(self, dataset_path, extension, target_size=None):
        """
        :param dataset_path: emplacement du data set qui doit être un repertoire non compressé et il doit comprendre
        des sous repertoires qui sont les classes des images. Le nom de ces sous dossiers serviront de label aux images
        qu'ils contiennent.
        :param extension: format des images du data set ex : png, jpg, jpeg ...
        :param target_size: tuple (hauteur largeur) désiré pour les images. Si None, elles ne seront pas redimensionnées
        """
        self.labels = []  # liste des labels possibles
        self.images = []  # vecteur des images
        self.images_labels = []  # vecteur des labels

        self._load_images(dataset_path, extension, target_size)

        self.images = np.array(self.images, dtype='float32')
        self.images_labels = np.array(self.images_labels)

    def get_X(self):
        """
        :return: une matrice numpy avec sur chaque ligne une matrice numpy qui est la représentation de l'image
        """
        return self.images

    def get_Y(self):
        """
        :return: un vecteur numpy représentant les labels des images
        soit un entier correspondant à la position du label retourné par DataSet.get_labels()
        soit la transformation effectué par DataSet.categorize_Y()
        """
        return self.images_labels

    def get_labels(self):
        """
        :return: la listes des labels possibles
        """
        return self.labels

    def number_classes(self):
        """
        :return: nombre de classe d'images présente dans le dataset
        """
        return len(self.labels)

    def categorize_Y(self):
        """
        Catégorise Y, pour que chaque label soit composé de DataSet.number_classes() bit au lieu de l'index de son label
        :return: None
        """
        self.images_labels = to_categorical(self.images_labels, self.number_classes())

    def normalize_X(self):
        """
        Mets les valeurs des images entre 0 et 1 au lieu de 0 et 255
        :return: None
        """
        self.images /= 255

    def _load_images(self, dataset_path, extension, target_size):
        previous_label = ""
        dir_elem, path_symbol = _get_dir_elem(dataset_path, extension)
        for img_path in dir_elem:
            # le label est le nom du dernier repertoire composant le chemin
            i = img_path.rfind(path_symbol)
            j = img_path.rfind(path_symbol, 0, i)
            actual_label = img_path[j + 1: i]  # label de l'image actuelle

            if actual_label != previous_label:  # quand on changera de répertoire le label changera
                self.labels.append(actual_label)  # il y a donc une nouvelle classe
            actual_label_index = len(self.labels) - 1  # indice du label dans self.labels
            
            self.images.append(image.img_to_array(image.load_img(img_path, target_size=target_size)))
            self.images_labels.append(actual_label_index)

            previous_label = actual_label  # label de la précédente image
