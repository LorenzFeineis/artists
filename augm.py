import read_data
import argparse
from PIL import Image
import numpy as np
import os, errno

parser = argparse.ArgumentParser(description='arguments for classification')
parser.add_argument('--aug_images', type=str, default='aug_images', metavar='X', help='image folder')
parser.add_argument('--scale', type=list, default=(200, 200), metavar='X', help='input layer size')
parser.add_argument('--img_process', type=str, default='crop', metavar='X', help='determines how images may be processed')
parser.add_argument('--crops', type=int, default=4, metavar='X', help='number of crops being taken if img_process = crop')
parser.add_argument('--color_convert', type=str, default="RGB", metavar='X', help='number of crops being taken if img_process = crop')
parser.add_argument('--percent_l2_norm', type=float, default=0.3, metavar='X', help='L2 Norm on generated images and origin. Only take this percent range')
args = parser.parse_args()


def dist_images(img1, img2):
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    dist = np.linalg.norm(img1 - img2)
    return dist

def flip_v_img(img):
    img_flip = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    dist = dist_images(img, img_flip)
    return img_flip, dist

def flip_h_img(img):
    img_flip = img.transpose(method=Image.FLIP_TOP_BOTTOM)
    dist = dist_images(img, img_flip)
    return img_flip, dist

def enhance_dataset(dataset):
    data = []
    for img, label in dataset:
        img_flip, dist = flip_v_img(img)
        data.append([img_flip, label, dist])

        img_flip2, dist2 = flip_h_img(img)
        data.append([img_flip2, label, dist2])

    k = np.uint32(np.round(np.shape(data)[0] * args.percent_l2_norm, 0))
    data = np.asarray(data)
    data = data[data[:, 2].argsort()][:k,:2]
    dataset = np.append(dataset, data)

    return dataset

def img_resize(img):
    return img.resize(args.scale)

def augment_image(img):
    return img_resize(img)

def rand_crops(img):
    w, h = img.size
    w_args, h_args = args.scale

    w -= w_args
    h -= w_args

    rand_w = np.random.randint(0, w, args.crops)
    rand_h = np.random.randint(0, h, args.crops)

    crop_list = []
    for width, height in zip(rand_w, rand_h):
        crop_list.append(img.crop((width, height, (width + w_args), (height + h_args))))

    return crop_list

def augment_dataset(dataset):
    dataset = transform_color(dataset)
    count = np.shape(dataset)[0]

    if args.img_process == 'crop':
        dataset2 = []
        for idx in range(count):
            crops = rand_crops(dataset[idx, :][0])
            for img in crops:
                dataset2.append([img, dataset[idx, :][1]])
        dataset = dataset2

    if args.img_process == 'resize':
        for idx in range(count):
            dataset[idx, :][0] = augment_image(dataset[idx, :][0])

    return dataset

def transform_color(dataset):
    if args.color_convert == "RGB":
        for idx, data in enumerate(dataset):
            img, label = data
            if img.getcolors() != None:
                dataset[idx, :][0] = img.convert('RGB')
        return dataset

    if args.color_convert == "L":
        for idx, data in enumerate(dataset):
            img, label = data
            if img.getcolors() == None:
                dataset[idx, :][0] = img.convert('L')
        return dataset


def save_augmentation(dataset, id2artist):
    try:
        os.makedirs(args.aug_images)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    i = 0
    ii = 0
    while i < dataset.size:
        img = dataset[i]
        i +=1
        label = dataset[i]
        i += 1
        name = id2artist[np.argmax(label)]
        ii += 1
        name = str(name) + str(ii) + ".jpg"
        img.save(os.path.join(args.aug_images, name))

def transform_format(dataset):
    dataset2 = []
    i = 0

    while i < dataset.size:
        img = dataset[i]
        i +=1
        label = dataset[i]
        i += 1
        dataset2.append([img, label])

    return dataset2

def get_augmentation(dataset, id2artist):
    dataset = augment_dataset(dataset)
    dataset = enhance_dataset(dataset)
    save_augmentation(dataset, id2artist)
    dataset = transform_format(dataset)

    return dataset

if __name__ == '__main__':
    dataset, id2artist = read_data.load_images()
    dataset = augment_dataset(dataset)
    dataset = enhance_dataset(dataset)
    save_augmentation(dataset, id2artist)
    dataset = transform_format(dataset)




