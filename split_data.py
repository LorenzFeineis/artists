### Create two folders one for training_images, one for test_images
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import os

root_dir = "aug_images/"
csv_file = "artists_changed.csv"


artists_info = pd.read_csv(csv_file)

artists_idx = [8,15,20,30]

artists_ids = range(len(artists_idx))
artists_names = [artists_info["name"][idx] for idx in artists_idx]
artists2id = dict(zip(artists_names,artists_ids))

image_list = []
for name in artists2id.keys():
    file_start = name.replace(" ","_")
    for file in os.listdir(root_dir):
        if file.startswith(file_start):
            image_list.append(file)

train_data, test_data = train_test_split(image_list, train_size = 0.9, random_state=42)

for train_file in tqdm(train_data):
    img = Image.open("aug_images/"+train_file)
    img.save("train_set/{}".format(train_file))
    img.close()
print("trainings data saved to train_set/")

for test_file in tqdm(test_data):
    img = Image.open("aug_images/"+test_file)
    img.save("test_set/{}".format(test_file))
    img.close()
print("test data saved to test_set/")
