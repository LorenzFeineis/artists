from augm import get_augmentation, rand_crops
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

import argparse
import numpy as np
import errno

parser = argparse.ArgumentParser(description='arguments for classification')
parser.add_argument('--aug_images', type=str, default='../aug_images', metavar='X', help='image folder')
parser.add_argument('--scale', type=list, default=(256, 256), metavar='X', help='input layer size')
parser.add_argument('--img_process', type=str, default='crop', metavar='X', help='determines how images may be processed')
parser.add_argument('--crops', type=int, default=4, metavar='X', help='number of crops being taken if img_process = crop')
parser.add_argument('--color_convert', type=str, default="RGB", metavar='X', help='number of crops being taken if img_process = crop')
parser.add_argument('--percent_l2_norm', type=float, default=1, metavar='X', help='L2 Norm on generated images and origin. Only take this percent range')
args = parser.parse_args()

artists_info = pd.read_csv('artists.csv')


os.chdir('resized') ### working directory is now /resized
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

index = 0
for image in tqdm(os.listdir()):

    index +=1
    parts = image.split("_")
    artist = parts[0]
    for part in parts[1:-1]:
        artist += " " + part
    if artist[0:8] == "Albrecht":
        artist = "Albrecht Duerer"
    try:
        image_jpg = Image.open(image)


        w,h = image_jpg.size
        w_crop, h_crop = args.scale
        if w<=w_crop or h<=h_crop:
            image_jpg = image_jpg.resize(args.scale)
            image_list = [image_jpg]
        else:
            image_list = rand_crops(image_jpg)


        if args.color_convert == "RGB":
            if image_jpg.getcolors() != None:
                image_jpg = image_jpg.convert('RGB')

        elif args.color_convert == "L":
            if image_jpg.getcolors() == None:
                image_jpg = image_jpg.convert('L')



        for img in image_list:
            vertical = img.transpose(method=Image.FLIP_TOP_BOTTOM)
            horizontal = img.transpose(method=Image.FLIP_LEFT_RIGHT)
            name = artist.replace(" ", "_")
            for save_it in (img,vertical,horizontal):
                file_name = str(name) +"_"+ str(index) + ".jpg"
                save_it.save("../aug_images/"+file_name)
                index +=1

        image_jpg.close()

    except KeyError:
        print(artist,"not found in dictionary")
