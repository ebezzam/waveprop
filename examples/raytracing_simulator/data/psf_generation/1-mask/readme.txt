Generate a new PSF using a mask :
python ../../../../scripts/simulator/generate_psf.py --input_fp=mask.png --output_fp=result.npy --mode=mask

Export the PSF to tiff images to display them :
python ../../../../scripts/conversion/npy_to_tiff.py result.npy

(Mask credits : Julien Sahli)