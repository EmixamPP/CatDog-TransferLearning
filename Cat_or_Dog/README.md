# Chien ou chat ?

Vous trouverez ci-dessous les instructions et détails sur le programme
"Chien ou chat?". Le but de se programme est d'utiliser le transfer learning pour  réussir à déterminer 
si une image contient un chien ou un chat.

Ce programme utilise le deep learning et notamment les réseaux de neurones
convolutionels (CNN), grace à la librairie tensorflow/keras.

## Installation

Ce projet peut être build en utilisant:
```bash
poetry install
```

## Utilisation

Vous pouvez ensuite lancer le jeu, dans l'environnement virtuel nouvellement
créé, en utilisant la commande:

### train.py

Le fichier `train.py` permets d'entrainer un modèle sur un dataset donné en utilisant ou non le transfer learning.

Pour plus de détails:

```
CNN Trainer for the Cat or Dog app.

options:
  -h, --help            show this help message and exit
  -f FOLDER, --folder FOLDER
                        Destination folder to save the model after training ends.
  -d DATA, --data DATA  Images folder
  -m PRETRAINEDMODEL, --pretrainedmodel PRETRAINEDMODEL
                        Name of a pretrained network
  -mp MODELPATH, --modelpath MODELPATH
                        Path to an existing model
  -c CATEGORIES [CATEGORIES ...], --categories CATEGORIES [CATEGORIES ...]
                        Name of the categories
  -tl TRANSFERLEARNING, --transferlearning TRANSFERLEARNING
                        Is TL applied
  --train_set_size TRAIN_SET_SIZE, -trs TRAIN_SET_SIZE
                        size of the training set (if not given, tasks all the data possible)
  --test_set_size TEST_SET_SIZE, -ts TEST_SET_SIZE
                        size of the testing set
  --val_set_size VAL_SET_SIZE, -vs VAL_SET_SIZE
                        size of the validation set
  --n_epochs N_EPOCHS, -ne N_EPOCHS
                        number of epochs
```

### experiment.py

Le fichier `experiment.py` exécute toutes les expériences réalisées.

Pour plus d'informations:
```
Experiements on transfer learning.

options:
  -h, --help            show this help message and exit
  -f FOLDER, --folder FOLDER
                        Destination folder to save the model after training ends
  -d DATA, --data DATA  Images folder for the base model
  -tld TLDATA, --tldata TLDATA
                        Images folder for the transfer learning model
```

### plot.py

Le fichier `plot.py` produit tout les graphes de résultat où le premier argument est le dossier résultat produit au préalable.

