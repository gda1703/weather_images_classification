# -*- coding: utf-8 -*-

# Import des libraries
# Importation des libraries 
import glob
import numpy as np
import os, shutil
from PIL import Image
import random
from shutil import copyfile
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Libraries pour la modèlisation
from tensorflow import keras


# Fonction qui permet la lecture des iamges 
def GetImagesFromFolder(PATH,Class_Folder,ext):
    """
    Cette fonction peut etre utilisée pour importer les images et elle permet de redimensionner les images en entrée
    
    Args:
    -----------
        PATH: chemin d’accès au répertoire contenant les fichiers image.
        Class_Folder: Nom du sous-répertoire contenant les fichiers image.
        xxt: SExtension de fichier des fichiers image à charger.

    Returns:
        Une liste d’images redimensionnées et un tableau NumPy de noms de classe correspondants.
        
    """
    
    images = [Image.open(file).convert('RGB').resize((224, 224),resample=Image.LANCZOS) for e in ext for file in glob.glob(PATH+Class_Folder+'/*.' + e)] 
    print(f"Dans le répertoire {Class_Folder} il y a {len(images)}")
    np.random.shuffle(images)
    return images,np.array([Class_Folder for i in range(len(images))])


# Sauvegarder les images des différents dataset dans le bon repertoire en local 
def save_images(files, labels, directory):
    """
    Enregistre une liste d’images avec les étiquettes de classe correspondantes dans un répertoire.

    Args:
    -----------
        files: liste d’objets Image représentant les images à enregistrer.
        labels : un tableau NumPy d’étiquettes de classe correspondantes pour chaque image dans 'files'.
        directory: chemin d’accès du répertoire pour enregistrer les images.

    Returns:
        Aucun
    
    """
    for idx, img in enumerate(files):
        label = labels[idx]
        filename = f"{idx}.jpg"
        filepath = os.path.join(directory, label, filename)
        if not os.path.exists(filepath):
            img.save(filepath)
            
            
# Fonction qui sauvarge en pickle le modèle 
def save_model(model, filename):
    """
    Sauvegarde du modele donné au format pickle si cela n'a pas déja été fait

    Args:
    -----------
       - model (object): le modèle qu'on veut sauvgarder
       - filename (str): le nom dy fichier pickle pour la sauvgarde

    Returns: 
        Aucun
        
    """
    if not os.path.isfile(filename):
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
            
            
# Fonction qui affiche le graphique qui affiche l'accuracy et la loss du modèle pour le train et la validation 
def plot_history(history):
    """
    Trace la précision et les valeurs de perte d’un modèle Keras sur chaque époque sur les ensembles d’apprentissage et de validation.

    Args:
    -----------
        history : objet history renvoyé par la méthode fit d’un modèle Keras pendant l’entraînement.

    Returns: 
        Aucun
    
    """
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].plot(history.history['accuracy'])
    axs[0].plot(history.history['val_accuracy'])
    axs[0].set_title('Accuracy du modèle')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'validation'], loc='upper left')

    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_title('Fonction de perte du modèle')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'validation'], loc='upper left')

    plt.show()
    
    
            

# Fonction qui permet d'evaluer le modèle 
def evaluate_model(model, train_generator, validation_generator):
    """
    Évalue un modèle donné sur les données d'entraînement et de validation fournies par l'utilisateur.
    
    Args :
    -----------
        - modèle : le modèle à évaluer
        - train_generator : le générateur de données d'entraînement
        - validation_generator : le générateur de données de validation
    
    Returns:
        - Un tuple contenant la fonction de perte d'apprentissage, la précision d'apprentissage, 
            la fonction de perte de validation et la précision de validation
    """
    loss_train, accuracy_train = model.evaluate(train_generator, verbose=0)
    loss_val, accuracy_val = model.evaluate(validation_generator, verbose=0)
    print(f"Validation dataset: Loss = {loss_val:.4f}, Accuracy = {accuracy_val:.4f}")
    print(f"Train dataset: Loss = {loss_train:.4f}, Accuracy = {accuracy_train:.4f}")
    





