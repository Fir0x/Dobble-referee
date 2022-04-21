#!/usr/bin/env python
# coding: utf-8

import os.path
from os import path
import requests
import tarfile
import shutil

PATH_TO_RESOURCES = '../resources'

# ## Fonction init_resources
#
# **param force** : 1 si l'on veut télécharger la donnée m
#
# 1. Controlle l'existence d'un dossier de **resources** situé à la racine du
#    projet
# 2. Si ce dossier existe et que **force == 0** alors il n'y a rien à faire
# 3. Sinon, le dossier est crée, l'archive est téléchargée et décompréssée puis
#    supprimée.


# Check if a local resources file exists.
# If not or force=1, creates the files and uploads data.
def init_resources(force=0):
    if path.exists(PATH_TO_RESOURCES) and path.isdir(PATH_TO_RESOURCES):
        if not force:
            print("Folder 'resources' found.")
            return
        else:
            shutil.rmtree(PATH_TO_RESOURCES)

    # Create folder
    os.mkdir(PATH_TO_RESOURCES)
    print("[1/4] Folder '../resources' created.")

    # Download data
    url_data = 'https://www.lrde.epita.fr/~jchazalo/SHARE/dobble-symbols-dataset-train.tar.gz'
    url_dobble_img = 'https://boardgamereview.co.uk/wp-content/uploads/2020/02/Dobble-cards-pile-1.png'
    tar_path = os.path.join(PATH_TO_RESOURCES,
                            'dobble-symbols-dataset-train.tar.gz')
    img_path = os.path.join(PATH_TO_RESOURCES, 'dobble_ref.png')
    response = requests.get(url_data, stream=True)
    if response.status_code == 200:
        with open(tar_path, 'wb') as f:
            f.write(response.raw.read())
    response = requests.get(url_dobble_img, stream=True)
    if response.status_code == 200:
        with open(img_path, 'wb') as f:
            f.write(response.raw.read())
            print('[2/4] Resources files uploaded.')

    # Extract from tar
    tar = tarfile.open(tar_path)
    tar.extractall(PATH_TO_RESOURCES)  # specify which folder to extract to
    tar.close()
    print('[3/4] Resources files extracted.')

    # Delete dowloaded tar
    os.remove(tar_path)
    print('[4/4] Tarball removed.')


# Call init_resources
init_resources(1)
