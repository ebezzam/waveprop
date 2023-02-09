# waveprop: Diffraction-based wave propagation simulator with PyTorch support

Python simulator for optical wave propagation based on scalar diffraction theory. Multiple propagation models are 
supported, with the desired propagation distance / complexity determining which one may be best. PyTorch support enables GPU acceleration
and end-to-end training of arbitrary apertures.

![lcav](/data/lcav.gif)

## Features

- Multiple scalar diffraction models: Fraunhofer, Fresnel, angular spectrum method, direct integration.
- Polychromatic through CIE color matching functions.
- Off-axis propagation and rescaling.
- PyTorch support (for GPU acceleration and end-to-end training).
- Arbitrary amplitude or phase masks.
- Spatial light modulator (SLM) simulator which incorporates deadspace and color filter.

## Installation

```sh
pip install waveprop
```

To develop locally and/or play with examples, we recommend the following steps:
```sh
# create virtual environment
conda create -n waveprop python=3.9
conda activate waveprop

# install
pip install -e .

# for CUDA, check docs for appropriate command: https://pytorch.org/
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

# for some examples (e.g. holography.py)
pip install joblib imageio click

# run tests
pytest tests/
```

# New release and upload to PyPi

From master branch of original repo, and using the appropriate value for `X.X.X`:

```
# Create tag and upload
git tag -a vX.X.X -m "Description."
git push origin vX.X.X

# Change version number in setup.py
python setup.py sdist

# Create package and upload to Pypi
twine upload dist/waveprop-X.X.X.tar.gz   
```

You will need a username and password for uploading to PyPi.

Finally, [on GitHub](https://github.com/ebezzam/waveprop/tags) set the new tag as the latest release by pressing the three dots to the right and selecting "Edit release, at top right selecting "Edit tag", and then publishing it!

## Examples

In the [`examples`] folder are various scripts demonstrating the features of `waveprop`. It is recommended to run them from the repository root, as shown below.

#### Comparing propagation models

#### Polychromatic simulation

#### Off-axis and rescaling

#### PyTorch support

#### Spatial light modulator

#### Holography

The above GIF showing the propagation of a holography pattern was generated with the following command:

```
python examples/holography.py --target data/lcav.png --invert
```

The file path can be set to any local path, however the target will be reshaped to a square.

If only interested in the holography pattern at a single distance, e.g. the focal plane, the following command can be run, which will produce a GIF with a single image

```
python examples/holography.py --target data/lcav.png --invert --f_lens 0.5 --z_start 0.5 --nz 1
```


Scripts and functions to simulate free-space optical propagation. 

In the `examples` folder:
- `holography.py`: determing phase pattern for holography and propagating over distances with angular spectrum method.
- `adafruit_slm.py`: polychromatric simulation of amplitude SLM with or without deadspace.
- `adafruit_slm_mono_pytorch.py`: monochromatric simulation of amplitude SLM with PyTorch support.
- `square_ap_video.py`: to compare various propagation approaches while varying the distance.
- `square_ap_poly_video.py`: polychromatic simulation of square aperture while varying the distance.
- `circ_ap_fraunhofer.py`: simulate circular aperture in the Fraunhofer regime.
- `square_ap_fresnel.py`: simulate square aperture in the Fresnel regime.
- `bandlimiting_angular_spectrum.py`: show benefit of band-limiting angular spectrum method.
- `off_axis.py`: comparing off-axis simulation with Fresnel, angular spectrum, and direct integration.
- `rescale.py`: comparing off-axis, rescaled simulation with Fresnel and angular spectrum.
- `circ_ap_lab.py`: simulate circular aperture with command-line defined arguments. Default is our lab setup.
- `rect_ap_lab.py`: simulate rectangular aperture with command-line defined arguments. Default is our lab setup.
- `single_slit_lab.py` (WIP): simulate single-slit with command-line defined arguments. Default is our lab setup.

NB: `click` is required for some scripts for parsing command-line arguments.

Following propagation models are implemented. All make use of FFT unless otherwise noted.
- Fraunhofer.
- Fresnel (one-step, two-step, multi-step, angular spectrum).
- Angular spectrum, with evanescent waves and option to bandlimit.
- Direct integration (no FFT), "brute force" numerical integration.
- FFT-DI, linearizes circular convolution of direction integration in DFT domain.
- Shifted Fresnel, uses three-FFT to model propagation off of optical axis with arbitrary input and
output sampling.
  
Note that dimensions `y` corresponds to the first dimension (rows) while `x`
corresponds to the second dimension (columns).

## Literature and references

Fraunhofer and Fresnel numerical approaches come from the textbook ["Numerical Simulation of Optical 
Wave Propagation with Examples in MATLAB" (2010)](https://www.spiedigitallibrary.org/ebooks/PM/Numerical-Simulation-of-Optical-Wave-Propagation-with-Examples-in-MATLAB/eISBN-9780819483270/10.1117/3.866274?SSO=1).
It is a very nicely-written book with code examples that are easy to follow. 

A more rigorous treatment and derivation of Fraunhofer and Fresnel approximations analytic 
expressions, and conditions can be found in the following textbooks (we reference the following
versions in the docstrings):
- "Introduction to Fourier Optics" by Goodman (Second Edition).
- "Fundamentals of Photonics" by Saleh and Teich (Third Edition).
- "Principles of Optics" by Born and Wolf (Seventh Edition).

A description of the Direct Integration (DI) method and its FFT version can be found in 
["Fast-Fourier-transform based numerical integration method for the Rayleigh–Sommerfeld diffraction 
formula" (2006)](https://www.osapublishing.org/ao/fulltext.cfm?uri=ao-45-6-1102&id=87971). This 
serves as a good baseline as it is an approximation of the Rayleigh-Sommerfeld diffraction integral 
via a Riemann sum (Eq 9). Main drawbacks are computational load, as DI directly performs the 
discrete convolution and FFT-DI requires three FFTs. Moreover, FFT-DI is only practical for small
output windows.

The angular spectrum (AS) approach is another well-known formulation that is directly derived from 
the Rayleigh-Sommerfeld equation. However, it tends to have issues outside of the scenarios of
small apertures and in the near-field. ["Band-Limited Angular Spectrum Method for
Numerical Simulation of Free-Space Propagation in Far and Near Fields"](https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-17-22-19662&id=186848)
offers a solution to this problem by limiting the bandwidth of the propagation field. An 
implementation of this approach has been largely replicated from [the code](https://github.com/computational-imaging/neural-holography/blob/d2e399014aa80844edffd98bca34d2df80a69c84/propagation_ASM.py#L22)
of ["Neural holography with camera-in-the-loop training"](https://dl.acm.org/doi/abs/10.1145/3414685.3417802). 
Their index-to-frequency mapping and FFT shifting seemed to be off, and they did not include evanescent waves; both of which were modified for the implementation found here.

Shifted Fresnel allows for the simulation off of the optical-axis and for arbitrary input and output
sampling. A description of this approach can be found in ["Shifted Fresnel diffraction for 
computational holography"](https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-15-9-5631&id=132698).

## TODO

Propagation models:
- Interpolation of angular spectrum with FS coefficients (ours).
- Rectangular tiling, Section 4 of "Shifted Fresnel diffraction for computational holography".
- Circ aperture and single slit exp from "Fast-Fourier-transform based numerical integration
method for the Rayleigh–Sommerfeld diffraction formula"

Examples:
- For single slit, something terribly wrong. Compare with following results:
    - https://core.ac.uk/download/pdf/233057112.pdf
    - https://www.osapublishing.org/josa/fulltext.cfm?uri=josa-59-3-293&id=53644
    - https://www.osapublishing.org/josa/fulltext.cfm?uri=josa-59-3-293&id=53644
- Double slit

Analytic forms of (see Born book):
- Fresnel circular aperture
- Fresnel rectangular aperture
- Circular aperture from "Fast-Fourier-transform based numerical integration
method for the Rayleigh–Sommerfeld diffraction formula", Eq 36

Compare / mention complexity of different approaches

## Other libraries

- Diffractio: https://diffractio.readthedocs.io/en/latest/readme.html
    - Cite "Applied Optics" vol 45 num 6 pp. 1102-1110 (2006) for Rayleigh Sommerfeld propagation.
- LightPipes: https://opticspy.github.io/lightpipes/manual.html#free-space-propagation
- AOtools: https://aotools.readthedocs.io/en/v1.0.1/opticalpropagation.html
    - Ported scripts from "Numerical Simulation of Optical Wave Propagation with Examples in MATLAB".
- PyOptica: https://gitlab.com/pyoptica/pyoptica
    - Free-space propagation: https://gitlab.com/pyoptica/pyoptica/-/blob/master/pyoptica/optical_elements/free_space.py
    - Gerschberg-Saxton: https://gitlab.com/pyoptica/pyoptica/-/blob/master/notebooks/gerchberg_saxton.ipynb
- DeepOptics: https://github.com/vsitzmann/deepoptics
    - Differentiable free-space propagation: https://github.com/vsitzmann/deepoptics/blob/defbb975309a6a3f3d2a86b92e82d02156ab213e/src/layers/optics.py#L386
- Angular Spectum: https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method
    - Monochromatic: https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method/blob/5b3610643b97ab6b81c80ef4c8aa5b0d9501f314/diffractsim/monochromatic_simulator.py#L191
    - Polychromatic: https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method/blob/5b3610643b97ab6b81c80ef4c8aa5b0d9501f314/diffractsim/polychromatic_simulator.py#L190
  

## License

MIT