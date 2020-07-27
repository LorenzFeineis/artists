import numpy as np
import torch
import numpy as np
from PIL import Image

import torchvision
from torchvision import transforms, utils

from create_dataset import ArtistsPaintingsDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from NeuralNet import Net

from sklearn.model_selection import cross_val_predict
from skorch import NeuralNetClassifier
from sklearn.model_selection import cross_val_score

### Transforms the images from the Dataset into torch.tensors()
transform = transforms.Compose([transforms.ToTensor()])

### Indixes of the chosen artists in artists_changed.csv
artists_idx = [8,13,15,20,30,31,32,46]

### The Train Dataset
train_data = ArtistsPaintingsDataset(root_dir="aug_images/", transform = transform, mode="Train", artists_idx=artists_idx)

### adjusting the tensors for the cross validation
x_cv, y_cv = train_data.samples, train_data.targets
x_cv = torch.cat(x_cv, dim = 0).resize(len(y_cv),3,256,256)
print(x_cv.shape)
y_cv = torch.tensor(y_cv)
print(y_cv.shape)

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)
print(device)

### will be a list of ouputs from cross_val_score
scores = []

### (1e-5,... ,1e1)
for lr in np.logspace(-5,1,7):
    ### net is the classifier based on our architecture with changing learning rate
    net = NeuralNetClassifier(Net(size= (256,256), num_classes = 8),
                             max_epochs = 100,
                             train_split=None,
                             criterion = torch.nn.CrossEntropyLoss,
                             optimizer = torch.optim.SGD,
                             lr = lr,
                             optimizer__momentum = 0.9,
                             batch_size = 160)
    ### score is a numpy area with the scores of the classifier on each of the
    ### 10 cross validation set
    score = cross_val_score(net.to(device), x_cv.to(device), y_cv.to(device), cv=10)
    scores.append(scores)

scores = np.array(scores)
np.save("Cross_Validation_scores.npy", scores)
