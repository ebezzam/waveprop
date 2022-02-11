import matplotlib.pyplot as plt
import torch
from waveprop.util import sample_points, plot2d
import numpy as np
from waveprop.spherical import spherical_prop
from waveprop.pytorch_util import compute_numpy_error
from waveprop.dataset_util import FlickrDataset, CIFAR10Dataset, MNISTDataset


target_dim = [3040, 4056]  # RPi sensor
d1 = 1.55e-6  # RPi sensor
downsample_factor = 16
idx = 50
source_distance = 3  # [m]
device = "cuda"  # "cpu" or "cuda"
dataset = "CIFAR"
mono = False
random_input_phase = False

if mono:
    wv = 640e-9  # red wavelength
else:
    wv = np.array([460, 550, 640]) * 1e-9

# downsample
target_dim = [target_dim[0] // downsample_factor, target_dim[1] // downsample_factor]
d1 *= downsample_factor

# load dataset
if dataset == "MNIST":
    """MNIST - 60'000 examples of 28x28"""
    ds = MNISTDataset(target_dim=target_dim, device=device, grayscale=mono)

elif dataset == "CIFAR":
    """CIFAR10 - 50;000 examples of 32x32"""
    ds = CIFAR10Dataset(target_dim=target_dim, device=device, grayscale=mono)

elif dataset == "FLICKR":
    """Flickr8k - varied, around 400x500"""
    ds = FlickrDataset(
        root_dir="/home/bezzam/Documents/Datasets/Flickr8k/images",
        captions_file="/home/bezzam/Documents/Datasets/Flickr8k/captions.txt",
        target_dim=target_dim,
        device=device,
        grayscale=mono,
    )

else:
    raise ValueError("Not supported dataset...")

print("size of dataset", len(ds))

# look at single example
input_image = ds[idx][0]
print("label : ", ds[idx][1])
print("shape : ", input_image.shape)
print("dtype : ", input_image.dtype)
print("maximum : ", input_image.max().item())
print("minimum : ", input_image.min().item())
x1, y1 = sample_points(N=input_image.shape[1:], delta=d1)
plot2d(x1.squeeze(), y1.squeeze(), input_image.cpu(), title="input")

# simulate input field by adding random phase
input_image = torch.sqrt(input_image)
if random_input_phase:
    phase = torch.exp(1j * torch.rand(input_image.shape) * 2 * np.pi).to(device)
    input_image = input_image * phase


""" PyTorch """
# propagate kernel
phase_shifts = spherical_prop(input_image, d1, wv, source_distance, return_psf=True)
print("spherical psf (pytorch)", phase_shifts.shape)
print("spherical psf (pytorch)", phase_shifts.dtype)
angle_np = np.angle(phase_shifts.detach().cpu().numpy())
plot2d(
    x1.squeeze(),
    y1.squeeze(),
    angle_np[2] if not mono else angle_np,
    title=f"{source_distance}m propagation kernel phase (pytorch), red",
)

# propagate
res = torch.square(torch.abs(spherical_prop(input_image, psf=phase_shifts)))
res_norm = res / res.max().item()
plot2d(
    x1.squeeze(),
    y1.squeeze(),
    res_norm.cpu().detach(),
    title=f"{source_distance}m propagation (pytorch)",
)

""" NumPy """
# convert to numpy
input_image_np = input_image.detach().cpu().numpy()
print("input", input_image.dtype)

# propagate kernel
phase_shifts_np = spherical_prop(input_image_np, d1, wv, source_distance, return_psf=True)
print("spherical psf (numpy)", phase_shifts_np.dtype)
plot2d(
    x1.squeeze(),
    y1.squeeze(),
    np.angle(phase_shifts_np[2] if not mono else phase_shifts_np),
    title=f"{source_distance}m propagation kernel phase (numpy), red",
)

# propagate
res_np = np.abs(spherical_prop(input_image_np, psf=phase_shifts_np)) ** 2
plot2d(
    x1.squeeze(),
    y1.squeeze(),
    res_np / res_np.max(),
    title=f"{source_distance}m propagation (numpy)",
)

""" Differences """
print("\n-- phase shift differences")
compute_numpy_error(phase_shifts, phase_shifts_np)

print("\n-- output differences")
compute_numpy_error(res, res_np)


plt.show()