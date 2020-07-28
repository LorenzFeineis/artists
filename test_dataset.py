from torchvision import transforms, utils
from dataset import ArtistsPaintingsDataset
from torch.utils.data import Dataset, DataLoader

transform = transforms.Compose([transforms.CenterCrop(size=(256,256)),
                                transforms.ToTensor()])


artists_idx = [8,13,15,16,19,20,22,30,31,32,46]

artists_idx = [1,2]

train_data = ArtistsPaintingsDataset(transform = transform, mode="Train", artists_idx=artists_idx)
test_data = ArtistsPaintingsDataset(transform = transform, mode="Test", artists_idx=artists_idx)
print(len(train_data), "train images loaded.")
print(len(test_data), "test images loaded.")

train_loader = DataLoader(train_data, batch_size=2)
test_loader = DataLoader(test_data)


for i,sample in enumerate(train_loader):
    if i ==0:
        print(sample)
        x,y = sample[0]
        print(x.shape)
        print(y.shape)
