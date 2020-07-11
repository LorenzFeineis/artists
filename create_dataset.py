import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from tqdm import tqdm

artists_info = pd.read_csv("../artists_changed.csv")

class ArtistsPaintingsDataset(Dataset):

    def __init__(self, csv_file="../artists_changed.csv",
            root_dir="../resized",
            num_paintings=0,
            transform=None,
            artists=None,
            artists_idx=None):
        self.artists_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        ###
        ### dictionary keys: artists name as string, key: id in range(0,num_artists)
        if artists and artists_idx: ### if artists and artists_idx is give use artists_idx
            artists_names = [self.artists_info["name"][idx] for idx in artists_idx]
            artists_ids = range(len(artists_idx))
            self.artists2id = dict(zip(artists_names,artists_ids))
        elif artists_idx:
            print("Here we are")
            artists_ids = range(len(artists_idx))
            artists_names = [self.artists_info["name"][idx] for idx in artists_idx]
            self.artists2id = dict(zip(artists_names,artists_ids))
        elif artists:
            artists_ids  = range(len(artists))
            self.artists2id = dict(zip(artists,artists_ids))


        self.image_list = []
        for name in self.artists2id.keys():
            file_start = name.replace(" ","_")
            for file in os.listdir(self.root_dir):
                if file.startswith(file_start):
                    self.image_list.append(file)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self,idx):
        image_list = self.image_list
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
            try:
                artist_id = self.artists2id[artist]

                sample = image
                if self.transform:
                    sample = self.transform(image)
                image.close()
                samples.append(sample)
                targets.append(artist_id)
            except KeyError:
                pass
        return list(zip(samples, targets))
