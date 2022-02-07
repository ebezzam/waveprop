import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from waveprop.util import sample_points, plot2d
import numpy as np
from waveprop.spherical import spherical_prop
from waveprop.pytorch_util import compute_numpy_error
from waveprop.dataset_util import FlickrDataset, CIFAR10Dataset


target_dim = [3040, 4056]  # RPi sensor
d1 = 1.55e-6  # RPi sensor
downsample_factor = 16
idx = 50
source_distance = 3  # [m]
wv = 640e-9  # red wavelength
device = "cuda"  # "cpu" or "cuda"
dataset = "FLICKR"
random_input_phase = True

# downsample
target_dim = [target_dim[0] // downsample_factor, target_dim[1] // downsample_factor]
d1 *= downsample_factor


# load dataset
if dataset == "MNIST":
    """MNIST - 60'000 examples of 28x28"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.RandomResizedCrop(target_dim, ratio=(1, 1), scale=(0.5, 1.0)),
        ]
    )
    ds = datasets.MNIST("../data", train=True, download=True, transform=transform)

elif dataset == "CIFAR":
    """CIFAR10 - 50;000 examples of 32x32"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.RandomResizedCrop(target_dim, ratio=(1, 1), scale=(0.5, 1.0)),
        ],
    )
    ds = CIFAR10Dataset(root="./data", train=True, download=True, transform=transform)

elif dataset == "FLICKR":
    """Flickr8k - varied, around 400x500"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.RandomResizedCrop(target_dim, ratio=(1, 1), scale=(0.5, 1.0)),
        ]
    )
    ds = FlickrDataset(
        root_dir="/home/bezzam/Documents/Datasets/Flickr8k/images",
        captions_file="/home/bezzam/Documents/Datasets/Flickr8k/captions.txt",
        transform=transform,
        grayscale=True,
    )

else:
    raise ValueError("Not supported dataset...")
print("size of dataset", len(ds))


# look at single example
input_image = ds[idx][0].squeeze()
print("label : ", ds[idx][1])
print("shape : ", input_image.shape)
x1, y1 = sample_points(N=input_image.shape[:2], delta=d1)
plot2d(x1.squeeze(), y1.squeeze(), input_image.cpu(), title="input")

# simulate input field by adding random phase
input_image = torch.sqrt(input_image)
if random_input_phase:
    phase = torch.exp(1j * torch.rand(input_image.shape) * 2 * np.pi)
    input_image = input_image * phase


""" PyTorch """
# propagate kernel
phase_shifts = spherical_prop(input_image, d1, wv, source_distance, return_psf=True)
print("spherical psf (pytorch)", phase_shifts.dtype)
plot2d(
    x1.squeeze(),
    y1.squeeze(),
    np.angle(phase_shifts.detach().cpu().numpy()),
    title=f"{source_distance}m propagation kernel phase (pytorch)",
)

# propagate
res = spherical_prop(input_image, psf=phase_shifts)
plot2d(
    x1.squeeze(),
    y1.squeeze(),
    np.abs(res.detach().cpu().numpy()) ** 2,
    title=f"{source_distance}m propagation (pytorch)",
)

""" NumPy """
# convert to numpy
input_image = input_image.to(device=device)
input_image_np = input_image.detach().cpu().numpy()
print("input", input_image.dtype)

# propagate kernel
phase_shifts_np = spherical_prop(input_image_np, d1, wv, source_distance, return_psf=True)
print("spherical psf (numpy)", phase_shifts_np.dtype)
plot2d(
    x1.squeeze(),
    y1.squeeze(),
    np.angle(phase_shifts_np) ** 2,
    title=f"{source_distance}m propagation kernel phase (numpy)",
)

# propagate
res_np = spherical_prop(input_image_np, psf=phase_shifts_np)
plot2d(
    x1.squeeze(),
    y1.squeeze(),
    np.abs(res_np) ** 2,
    title=f"{source_distance}m propagation (numpy)",
)

""" Differences """
print("\n-- phase shift differences")
compute_numpy_error(phase_shifts, phase_shifts_np)

print("\n-- output differences")
compute_numpy_error(res, res_np)


plt.show()
