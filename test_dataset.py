from torchvision import transforms, utils
from create_dataset import ArtistsPaintingsDataset
from torch.utils.data import Dataset, DataLoader

transform = transforms.Compose([transforms.CenterCrop(size=(64,64)),
                                transforms.Grayscale(num_output_channels=3),
                                transforms.ToTensor()])
data = ArtistsPaintingsDataset(transform = transform)



#example = data[0:10]
#print(example)

data_loader = DataLoader(data, batch_size=1000)

for i,sample in enumerate(data_loader):
    print(sample[0][0].shape)
    print(sample[0][1].shape)
