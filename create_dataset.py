import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from tqdm import tqdm

#artists_info = pd.read_csv("../artists_changed.csv")
#print(artists_info["name"][19])

class ArtistsPaintingsDataset(Dataset):

    def __init__(self, csv_file="../artists_changed.csv",
            root_dir="../resized",
            num_paintings=0,
            transform=None,
            artists=None,
            artists_idx=None):
        self.artists_info = pd.read_csv(csv_file)
        self.artist2id = dict(zip(self.artists_info["name"], self.artists_info["id"]))
        self.root_dir = root_dir
        self.transform = transform
        self.num_paintings = num_paintings
    def __len__(self):
        image_list = os.listdir(self.root_dir)
        return len(image_list)

    def __getitem__(self,idx):
        image_list = os.listdir(self.root_dir)
        samples = []
        targets = []
        if type(idx) != slice:
            img_names = [image_list[idx]]
        else:
            img_names = image_list[idx]
        for img_name in img_names:
            image = Image.open(self.root_dir+"/"+img_name)
            artist_list = img_name.split("_")
            artist = artist_list[0]
            for string in artist_list[1:-1]:
                artist += " "+string
            if artist[0:8] == "Albrecht":
                artist = "Albrecht Duerer"
            artist_id = self.artist2id[artist]

            sample = image
            if self.transform:
                sample = self.transform(image)
            image.close()

            samples.append(sample)
            targets.append(artist_id)
        return list(zip(samples, targets))
