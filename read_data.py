from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def load_images():
    artists_info = pd.read_csv('artists.csv')


    os.chdir('resized_small2') ### working directory is now /resized
    print(os.getcwd())

    print("# of pictures in folder resized:", len(os.listdir()))


    ### Create list of artists
    artists = []
    for artist in artists_info["name"]:
        artists.append(artist.replace("ü", "ue")) ### replace ü with u because of Dürer

    ### Create dictionary with keys: artists, values: id
    ### Create dictionary with keys: artists, values: id
    artist2id = dict(zip
                     (artists,artists_info["id"]))
    id2artist = dict(zip
                     (artists_info["id"], artists))

    print("Artists:", artist2id.keys())

    ### dataset (num_images,2) for each image dataset containes the image as .jpg
    ### and the the binary vector with a 1 at the correpsonding index for the artist.
    dataset = []
    label_vector = np.array([0 for artist in artists])
    for image in os.listdir():
        parts = image.split("_")
        artist = parts[0]
        for part in parts[1:-1]:
            artist += " " + part
        if artist[0:8] == "Albrecht":
            artist = "Albrecht Duerer"
        try:
            label = artist2id[artist]
            labeled_vec = label_vector.copy()
            labeled_vec[label] = 1
            image_jpg = Image.open(image)
            dataset.append([image_jpg, labeled_vec])
        except KeyError:
            print(artist,"not found in dictionary")

    dataset = np.array(dataset)
    return dataset, id2artist
