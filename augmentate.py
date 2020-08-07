from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

import argparse
import numpy as np
import errno

parser = argparse.ArgumentParser(description='arguments for augmentation')
parser.add_argument('--source', type=str, default='../resized', metavar='X', help='image folder')
parser.add_argument('--target_dir', type=str, default='../train_aug_images/', metavar='X', help='target folder')
parser.add_argument('--scale', type=list, default=(256, 256), metavar='X', help='input layer size')
parser.add_argument('--crops', type=int, default=4, metavar='X', help='number of crops being taken')
parser.add_argument('--color_convert', type=str, default="RGB", metavar='X', help='colorcode to convert images to')
args = parser.parse_args()

def rand_crops(img, crops=4, scale=(256, 256)):
    w, h = img.size
    w_args, h_args = scale

    w -= w_args
    h -= w_args

    rand_w = np.random.randint(0, w, crops )
    rand_h = np.random.randint(0, h, crops)

    crop_list = []
    for width, height in zip(rand_w, rand_h):
        crop_list.append(img.crop((width, height, (width + w_args), (height + h_args))))

    return crop_list


def augmentate(source,target,crops,scale,color_convert):
    artists_info = pd.read_csv('artists.csv')


    os.chdir(source) ### Change working directory

    print(os.getcwd())

    print("# of pictures in source directory:", len(os.listdir()))


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
            image_jpg = image_jpg.convert('RGB')

            w,h = image_jpg.size
            w_crop, h_crop = scale
            if w<=w_crop or h<=h_crop:
                image_jpg = image_jpg.resize(scale)
                image_list = [image_jpg]
            else:
                image_list = rand_crops(image_jpg,crops,scale)


            if color_convert == "RGB":
                if image_jpg.getcolors() != None:
                    image_jpg = image_jpg.convert('RGB')

            elif color_convert == "L":
                if image_jpg.getcolors() == None:
                    image_jpg = image_jpg.convert('L')



            for img in image_list:
                vertical = img.transpose(method=Image.FLIP_TOP_BOTTOM)
                horizontal = img.transpose(method=Image.FLIP_LEFT_RIGHT)
                name = artist.replace(" ", "_")
                for save_it in (img,vertical,horizontal):
                    file_name = str(name) +"_"+ str(index) + ".jpg"
                    save_it.save(target+"/"+file_name)
                    index +=1

            image_jpg.close()

        except KeyError:
            print(artist,"not found in dictionary")

if __name__ == '__main__':
    source = args.source
    target = args.target_dir
    crops = args.crops
    scale = args.scale
    color_convert = args.color_convert
    augmentate(source,target,crops,scale,color_convert)
