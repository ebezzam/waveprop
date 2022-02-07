import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from PIL import Image
import os


class CIFAR10Dataset(datasets.CIFAR10):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
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
        return res[0], self.classes[res[1]]


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, grayscale=False, device=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        self.grayscale = grayscale
        self.device = device

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
        if self.grayscale:
            img = transforms.functional.rgb_to_grayscale(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.device:
            img = img.to(device=self.device)

        return img, caption
