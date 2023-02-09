import matplotlib.pyplot as plt
import torch
from waveprop.util import sample_points, plot2d
import numpy as np
from waveprop.spherical import spherical_prop
from waveprop.pytorch_util import compute_numpy_error
from waveprop.dataset_util import load_dataset
from waveprop.devices import SensorOptions, SensorParam, sensor_dict


sensor_config = sensor_dict[SensorOptions.RPI_HQ.value]
downsample_factor = 6
idx = 50
scene2mask = 0.4  # [m]
mask2sensor = 0.004
object_height = 5e-2
device = "cuda"  # "cpu" or "cuda"
dataset = "MNIST"
mono = True
random_input_phase = False

if mono:
    wv = 640e-9  # red wavelength
else:
    wv = np.array([460, 550, 640]) * 1e-9


# downsample
target_dim = (sensor_config[SensorParam.SHAPE] // downsample_factor).astype(int)
d1 = sensor_config[SensorParam.PIXEL_SIZE] * downsample_factor

# load dataset
ds = load_dataset(
    dataset,
    scene2mask=scene2mask,
    mask2sensor=mask2sensor,
    sensor_dim=sensor_config[SensorParam.SIZE],
    object_height=object_height,
    target_dim=target_dim,
    device=device,
    grayscale=mono,
    vflip=True,
    # for Flickr8
    root_dir="/home/bezzam/Documents/Datasets/Flickr8k/images",
    captions_file="/home/bezzam/Documents/Datasets/Flickr8k/captions.txt",
)

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
phase_shifts = spherical_prop(u_in=input_image, d1=d1, wv=wv, dz=scene2mask, return_psf=True)
print("spherical psf (pytorch)", phase_shifts.shape)
print("spherical psf (pytorch)", phase_shifts.dtype)
angle_np = np.angle(phase_shifts.detach().cpu().numpy())
plot2d(
    x1.squeeze(),
    y1.squeeze(),
    angle_np[2] if not mono else angle_np,
    title=f"{scene2mask}m propagation kernel phase (pytorch), red",
)

# propagate
res = torch.square(torch.abs(spherical_prop(u_in=input_image, psf=phase_shifts)))
res_norm = res / res.max().item()
plot2d(
    x1.squeeze(),
    y1.squeeze(),
    res_norm.cpu().detach(),
    title=f"{scene2mask}m propagation (pytorch)",
)

""" NumPy """
# convert to numpy
input_image_np = input_image.detach().cpu().numpy()
print("input", input_image.dtype)

# propagate kernel
phase_shifts_np = spherical_prop(u_in=input_image_np, d1=d1, wv=wv, dz=scene2mask, return_psf=True)
print("spherical psf (numpy)", phase_shifts_np.dtype)
plot2d(
    x1.squeeze(),
    y1.squeeze(),
    np.angle(phase_shifts_np[2] if not mono else phase_shifts_np),
    title=f"{scene2mask}m propagation kernel phase (numpy), red",
)

# propagate
res_np = np.abs(spherical_prop(u_in=input_image_np, psf=phase_shifts_np)) ** 2
plot2d(
    x1.squeeze(),
    y1.squeeze(),
    res_np / res_np.max(),
    title=f"{scene2mask}m propagation (numpy)",
)

""" Differences """
print("\n-- phase shift differences")
compute_numpy_error(phase_shifts, phase_shifts_np)

print("\n-- output differences")
compute_numpy_error(res, res_np)


plt.show()
