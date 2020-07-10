from torchvision import transforms, utils
from create_dataset import ArtistsPaintingsDataset

transform = transforms.ToTensor()
data = ArtistsPaintingsDataset(transform=transform)



example = data[0:10]
print(len(example[0]))
