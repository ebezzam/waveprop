from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np
import glob
from waveprop.simulation import FarFieldSimulator
from abc import abstractmethod


class Datasets(object):
    MNIST = "MNIST"
    CIFAR10 = "CIFAR"
    FLICKR8k = "FLICKR"


# TODO : take into account FOV and offset


class SimulatedDataset(Dataset):
    """
    Abstract class for simulated datasets.
    """

    def __init__(
        self,
        transform_list=None,
        psf=None,
        target="original",
        random_vflip=False,
        random_hflip=False,
        random_rotate=False,
        **kwargs,
    ):
        """

        Parameters
        ----------
        transform_list : list of torchvision.transforms, optional
            List of transforms to apply to image, by default None.
        psf : np.ndarray, optional
            Point spread function, by default None.
        target : str, optional
            Target to return, by default "original".
            "original" : return propagated image and original image.
            "object_plane" : return propagated image and object plane.
            "label" : return propagated image and label.
        random_vflip : float, optional
            Probability of vertical flip, by default False.
        random_hflip : float, optional
            Probability of horizontal flip, by default False.
        random_rotate : float, optional
            Maximum angle of rotation, by default False.
        """

        self.target = target

        # random transforms
        self._transform = None
        if transform_list is None:
            transform_list = []
        if random_vflip:
            transform_list.append(transforms.RandomVerticalFlip(p=random_vflip))
        if random_hflip:
            transform_list.append(transforms.RandomHorizontalFlip(p=random_hflip))
        if random_rotate:
            transform_list.append(transforms.RandomRotation(random_rotate))
        if len(transform_list) > 0:
            self._transform = transforms.Compose(transform_list)

        # initialize simulator
        if psf is not None:
            if psf.shape[-1] <= 3:
                raise ValueError("Channel dimension should not be last.")
        self.sim = FarFieldSimulator(psf=psf, is_torch=True, **kwargs)

    @abstractmethod
    def get_image(self, index):
        raise NotImplementedError

    def __getitem__(self, index):

        # load image
        img, label = self.get_image(index)
        if self._transform is not None:
            img = self._transform(img)

        # propagate and return with desired output
        if self.target == "original":
            return self.sim.propagate(img), img
        elif self.target == "object_plane":
            return self.sim.propagate(img, return_object_plane=True)
        elif self.target == "label":
            return self.sim.propagate(img), label

    def __len__(self):
        return self.n_files


class SimulatedDatasetFolder(SimulatedDataset):
    """
    Dataset of propagated images from a folder of images.
    """

    def __init__(self, path, image_ext="jpg", n_files=None, **kwargs):
        """
        Parameters
        ----------
        path : str
            Path to folder of images.
        image_ext : str, optional
            Extension of images, by default "jpg".
        n_files : int, optional
            Number of files to load, by default load all.
        """

        self.path = path
        self._files = glob.glob(os.path.join(self.path, f"*.{image_ext}"))
        if n_files is None:
            self.n_files = len(self._files)
        else:
            self.n_files = n_files
            self._files = self._files[:n_files]

        # initialize simulator
        super(SimulatedDatasetFolder, self).__init__(
            transform_list=[transforms.ToTensor()], **kwargs
        )

    def get_image(self, index):
        img = Image.open(self._files[index])
        label = None
        return img, label


class SimulatedPytorchDataset(SimulatedDataset):
    """
    Dataset of propagated images from a torch Dataset.
    """

    def __init__(self, dataset, **kwargs):
        """
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to propagate.
        """

        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.n_files = len(dataset)

        # initialize simulator
        super(SimulatedPytorchDataset, self).__init__(**kwargs)

    def get_image(self, index):
        return self.dataset[index]


class MNISTDataset(datasets.MNIST):
    def __init__(
        self,
        object_height,
        scene2mask,
        mask2sensor,
        sensor_dim,
        target_dim,
        fov=None,
        offset=None,
        device=None,
        root="./data",
        train=True,
        download=True,
        vflip=True,
        grayscale=True,
        scale=(1, 1),
        **kwargs,
    ):
        """
        MNIST - 60'000 examples of 28x28

        Parameters
        ----------
        device : "cpu" or "gpu"
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
        transform_list = [np.array, transforms.ToTensor()]
        # transform_list = []
        if vflip:
            transform_list.append(transforms.RandomVerticalFlip(p=1.0))

        # scale image to desired height at object plane
        magnification = mask2sensor / scene2mask
        self.scene_dim = np.array(sensor_dim) / magnification
        object_height_pix = int(np.round(object_height / self.scene_dim[1] * target_dim[1]))
        scaling = object_height_pix / self.input_dim[1]
        object_dim = (np.round(self.input_dim * scaling)).astype(int).tolist()
        transform_list.append(transforms.RandomResizedCrop(object_dim, ratio=(1, 1), scale=scale))

        # pad rest with zeros
        padding = np.array(target_dim) - object_dim
        left = padding[1] // 2
        right = padding[1] - left
        top = padding[0] // 2
        bottom = padding[0] - top
        transform_list.append(transforms.Pad(padding=(left, top, right, bottom)))

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
        **kwargs,
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
        **kwargs,
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
        try:
            import pandas as pd
            self.df = pd.read_csv(captions_file)
        except ImportError:
            raise ImportError("Please install pandas to use this dataset")
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


def get_object_height_pix(object_height, mask2sensor, scene2mask, sensor_dim, target_dim):
    """
    Determine height of object in pixel when it reaches the sensor.

    Parameters
    ----------
    object_height
    mask2sensor
    scene2mask
    sensor_dim
    target_dim

    Returns
    -------

    """
    magnification = mask2sensor / scene2mask
    scene_dim = sensor_dim / magnification
    return int(np.round(object_height / scene_dim[1] * target_dim[1]))


def load_dataset(dataset_str, **kwargs):
    """
    Load one of available datasets.

    TODO : take into account FOV of mask / sensor.

    Parameters
    ----------
    dataset_str
    scene2mask : float
        Distance between scene and mask. TODO : could be variable?
    ds : float
        Sampling at sensor.
    fov : float
        Field of view of camera in degrees. Limits placement of object.
    kwargs

    Returns
    -------

    """
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
