# waveprop

Scripts and functions to simulate free-space optical propagation. In the `examples` folder:
- `circ_ap_fraunhofer.py`: simulate circular aperture in the Fraunhofer regime.
- `rect_ap_fresnel.py`: simulate square aperture in the Fresnel regime.
- `circ_ap_lab.py`: simulate circular aperture with command-line defined arguments. Default is our lab setup.
- `rect_ap_lab.py`: simulate rectangular aperture with command-line defined arguments. Default is our lab setup.
- `single_slit_lab.py`: simulate rectangular aperture with command-line defined arguments. Default is our lab setup.

NB: `click` is required for some of the scripts for parsing command-line arguments.

Following propagation models are implemented:
- Fraunhofer
- Fresnel (one-step, two-step, angular spectrum)
- Angular spectrum, with evanescent waves and option to bandlimit
- Direct integration

## Local install

```sh
# recommended to create virtual environment
virtualenv -p python3 waveprop_env
source waveprop_env/bin/activate

# install
pip install -e .
```

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

A description of the Direct Integration (DI) method and its FFT version (todo) can be found in 
["Fast-Fourier-transform based numerical integration method for the Rayleigh–Sommerfeld diffraction 
formula" (2006)](https://www.osapublishing.org/ao/fulltext.cfm?uri=ao-45-6-1102&id=87971). This 
serves as a good baseline as it is an approximation of the Rayleigh-Sommerfled diffraction integral
through via a Riemann sum (Eq 9).

The angular spectrum (AS) approach is another well-known formulation that is directly derived from 
the Rayleigh-Sommerfeld equation. However, it tends to have issues outside of the scenarios of
small apertures and in the near-field. ["Band-Limited Angular Spectrum Method for
Numerical Simulation of Free-Space Propagation in Far and Near Fields"](https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-17-22-19662&id=186848)
offers a solution to this problem by limiting the bandwidth of the propagation field. An 
implementation of this approach has been largely replicated from [the code](https://github.com/computational-imaging/neural-holography/blob/d2e399014aa80844edffd98bca34d2df80a69c84/propagation_ASM.py#L22)
of ["Neural holography with camera-in-the-loop training"](https://dl.acm.org/doi/abs/10.1145/3414685.3417802). 
Their index-to-frequency mapping and FFT shifting seemed to be off, and they did not include evanescent waves; both of which were modified for the implementation found here.

## TODO

Propagation models:
- FFT-DI, "Fast-Fourier-transform based numerical integration method for the Rayleigh–Sommerfeld diffraction 
formula"
- Interpolation of angular spectrum with FS coefficients
- Shifted Fresnel method, "Shifted Fresnel diffraction for computational holography"

Examples:
- For single slit, something terribly wrong. Compare with following results:
    - https://core.ac.uk/download/pdf/233057112.pdf
    - https://www.osapublishing.org/josa/fulltext.cfm?uri=josa-59-3-293&id=53644
    - https://www.osapublishing.org/josa/fulltext.cfm?uri=josa-59-3-293&id=53644
- Double slit

Analytic forms:
- Fresnel circular aperture
- Fresnel rectangular aperture

Compare / mention complexity of different approaches

## Other libraries

- Diffractio: https://diffractio.readthedocs.io/en/latest/readme.html
    - Cite "Applied Optics" vol 45 num 6 pp. 1102-1110 (2006) for Rayleigh Sommerfeld propagation.
- LightPiptes: https://opticspy.github.io/lightpipes/manual.html#free-space-propagation
- AOtools: https://aotools.readthedocs.io/en/v1.0.1/opticalpropagation.html
    - Ported scripts from "Numerical Simulation of Optical Wave Propagation with Examples in MATLAB".
- PyOptica: https://gitlab.com/pyoptica/pyoptica