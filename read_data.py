from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


artists_info = pd.read_csv('../datasets_130081_310927_artists.csv')

os.chdir('../resized')
print(os.getcwd())
print(len(os.listdir()))
test = Image.open(os.listdir()[0])
#test.show()
images = '../resized'

#print(example)
artists = []
for artist in artists_info["name"]:
    artists.append(artist.replace("ü", "u"))
artist2id = dict(zip(artists,artists_info["id"]))
print(artist2id.keys())


dataset = []
label_vector = np.array([0 for artist in artists])
for image in os.listdir():
    parts = image.split("_")
    artist = parts[0]
    for part in parts[1:-1]:
        artist += " " + part
    if artist[0:8] == "Albrecht":
        artist = "Albrecht Durer"
    try:
        label = artist2id[artist]
        labeled_vec = label_vector.copy()
        labeled_vec[label] = 1
        image_jpg = Image.open(image)
        dataset.append([image_jpg, labeled_vec])
        image_jpg.close()
    except KeyError:
        print(artist,"not found in dictionary")

dataset = np.array(dataset)
print(dataset.shape)
