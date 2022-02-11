import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np


class Datasets(object):
    MNIST = "MNIST"
    CIFAR10 = "CIFAR"
    FLICKR8k = "FLICKR"


# TODO : abstract parent class for Dataset, add distance for far-field propagation


class MNISTDataset(datasets.MNIST):
    def __init__(
        self,
        device=None,
        target_dim=None,
        pad=None,
        root="./data",
        train=True,
        download=True,
        vflip=True,
        grayscale=True,
        scale=(1, 1),
        **kwargs
    ):
        """
        MNIST - 60'000 examples of 28x28

        Parameters
        ----------
        device
        target_dim
        pad
        root
        train
        download
        vflip
        grayscale
        scale
        kwargs
        """

        self.input_dim = np.array([28, 28])
        transform_list = [transforms.ToTensor()]
        if vflip:
            transform_list.append(transforms.RandomVerticalFlip(p=1.0))
        if pad:
            padding = (pad * self.input_dim / 2).astype(np.int).tolist()
            transform_list.append(transforms.Pad(padding))
        if target_dim:
            transform_list.append(
                transforms.RandomResizedCrop(target_dim, ratio=(1, 1), scale=scale)
            )
        if not grayscale:
            transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        transform = transforms.Compose(transform_list)
        self.device = device
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):

        res = super().__getitem__(index)
        if self.device:
            img = res[0].to(device=self.device)
        else:
            img = res[0]
        return img, res[1]


class CIFAR10Dataset(datasets.CIFAR10):
    def __init__(
        self,
        device=None,
        pad=None,
        target_dim=None,
        grayscale=False,
        root="./data",
        train=True,
        scale=(1, 1),
        download=True,
        vflip=True,
        **kwargs
    ):
        """
        CIFAR10 - 50;000 examples of 32x32

        Parameters
        ----------
        device
        pad
        target_dim
        grayscale
        root
        train
        scale
        download
        vflip
        kwargs
        """

        self.input_dim = np.array([32, 32])
        transform_list = [transforms.ToTensor()]
        if vflip:
            transform_list.append(transforms.RandomVerticalFlip(p=1.0))
        if grayscale:
            transform_list.append(transforms.Grayscale())
        if pad:
            padding = (pad * self.input_dim / 2).astype(np.int).tolist()
            transform_list.append(transforms.Pad(padding))
        if target_dim:
            transform_list.append(
                transforms.RandomResizedCrop(target_dim, ratio=(1, 1), scale=scale)
            )
        transform = transforms.Compose(transform_list)
        self.device = device
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

    def __getitem__(self, index):

        res = super().__getitem__(index)
        if self.device:
            img = res[0].to(device=self.device)
        else:
            img = res[0]
        return img, self.classes[res[1]]


class FlickrDataset(Dataset):
    def __init__(
        self,
        root_dir,
        captions_file,
        pad=None,
        vflip=True,
        target_dim=None,
        grayscale=False,
        device=None,
        scale=(1, 1),
        **kwargs
    ):
        """
        Flickr8k - varied, around 400x500

        Download dataset from : https://www.kaggle.com/adityajn105/flickr8k

        Parameters
        ----------
        root_dir
        captions_file
        pad
        vflip
        target_dim
        grayscale
        device
        scale
        kwargs
        """

        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.device = device

        transform_list = []
        if vflip:
            transform_list.append(transforms.RandomVerticalFlip(p=1.0))
        if grayscale:
            transform_list.append(transforms.Grayscale())
        if target_dim:
            transform_list.append(
                transforms.RandomResizedCrop(target_dim, ratio=(1, 1), scale=scale)
            )
        self.transform = transforms.Compose(transform_list)
        self.pad = pad

        # Get img, caption columns
        self.df = self.df.set_index("image")
        self.df = self.df[~self.df.index.duplicated(keep="first")]  # multiple captions per image
        self.imgs = self.df.index
        self.captions = self.df["caption"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_id = self.imgs[index]
        caption = self.captions[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        img = transforms.ToTensor()(img)
        if self.pad:
            padding = (np.array(list(self.pad * img.shape[1:])) / 2).astype(np.int).tolist()
            padding = (padding[0], padding[1], padding[0], padding[0])
            img = F.pad(img, padding)

        if self.transform is not None:
            img = self.transform(img)

        if self.device:
            img = img.to(device=self.device)

        return img, caption


def load_dataset(dataset_str, **kwargs):
    dataset = None
    if dataset_str == Datasets.MNIST:
        """MNIST - 60'000 examples of 28x28"""
        dataset = MNISTDataset(**kwargs)
    elif dataset_str == Datasets.CIFAR10:
        """CIFAR10 - 50;000 examples of 32x32"""
        dataset = CIFAR10Dataset(**kwargs)
    elif dataset_str == Datasets.FLICKR8k:
        """Flickr8k - varied, around 400x500"""
        dataset = FlickrDataset(**kwargs)
    else:
        raise ValueError("Not supported dataset...")
    return dataset


if __name__ == "__main__":

    from waveprop.util import plot2d, sample_points
    import matplotlib.pyplot as plt
    from torchvision.utils import save_image

    idx = 50
    dataset = Datasets.FLICKR8k
    target_dim = None
    device = "cpu"
    input_pad = None
    grayscale = False
    vflip = True

    ds = load_dataset(
        dataset,
        target_dim=target_dim,
        device=device,
        pad=input_pad,
        grayscale=grayscale,
        vflip=vflip,
        # for Flickr, need to download: https://www.kaggle.com/adityajn105/flickr8k
        root_dir="/home/bezzam/Documents/Datasets/Flickr8k/images",
        captions_file="/home/bezzam/Documents/Datasets/Flickr8k/captions.txt",
    )

    input_image = ds[idx][0]
    print("\n-- Input image")
    print("label", ds[idx][1])
    print("shape", input_image.shape)
    print("dtype", input_image.dtype)

    # plot
    x1, y1 = sample_points(N=list(input_image.shape[1:]), delta=1)
    plot2d(x1.squeeze(), y1.squeeze(), input_image.cpu())
    save_image(transforms.RandomVerticalFlip(p=1.0)(input_image), "image.png")

    plt.show()
