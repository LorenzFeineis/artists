import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class ArtistsPaintingsDataset(Dataset):

    def __init__(self, transform=None, mode = "Train",artists_idx = [8,15,20,30]):
        self.transform = transform
        ###
        ### dictionary keys: artists name as string, key: id in range(0,num_artists)
        self.artists_info = pd.read_csv("artists_changed.csv")

        artists_ids = range(len(artists_idx))
        artists_names = [self.artists_info["name"][idx] for idx in artists_idx]
        self.artists2id = dict(zip(artists_names,artists_ids))

        if mode == "Train":
            self.root_dir = "train_aug_images/"
        elif mode == "Test":
            self.root_dir = "test_aug_images/"
        else:
            print("mode must be \"Train\" or \"Test\"")

        self.image_list = []
        for name in self.artists2id.keys():
            file_start = name.replace(" ","_")
            for file in os.listdir(self.root_dir):
                if file.startswith(file_start):
                    self.image_list.append(file)

        self.samples = []
        self.targets = []
        print("Load data:")
        for img_name in tqdm(self.image_list):
            image = Image.open(self.root_dir+"/"+img_name).convert('RGB')
            artist_list = img_name.split("_")
            artist = artist_list[0]
            for string in artist_list[1:-1]:
                artist += " "+string
            try:
                artist_id = self.artists2id[artist]

                sample = image
                if self.transform:
                    sample = self.transform(image)
                image.close()
                self.samples.append(sample)
                self.targets.append(artist_id)
            except KeyError:
                pass

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self,idx):
        if type(idx)!= slice:
            targets = [self.targets[idx]]
        else:
            targets = self.targets[idx]
        return self.samples[idx], targets
