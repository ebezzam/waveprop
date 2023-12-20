Generate a new PSF using a height map :
python ../../../../scripts/simulator/generate_psf.py --input_fp=heights.png --output_fp=result.npy --mode=height --save_normals

Export the PSF to tiff images to display them :
python ../../../../scripts/conversion/npy_to_tiff.py result.npy

Try to change the size of the Sobel kernel operator to a smaller value and make the image sharper (min is 3, default is 7)
python ../../../../scripts/simulator/generate_psf.py --input_fp=heights.png --output_fp=result_sharp.npy --mode=height --save_normals --sobel_size=3

Export the PSF to tiff images to display them :
python ../../../../scripts/conversion/npy_to_tiff.py result_sharp.npy

Try to change the size of the Sobel kernel operator to a bigger value and make the image smoother
python ../../../../scripts/simulator/generate_psf.py --input_fp=heights.png --output_fp=result_smooth.npy --mode=height --save_normals --sobel_size=23

Export the PSF to tiff images to display them :
python ../../../../scripts/conversion/npy_to_tiff.py result_smooth.npy


(Height map credits : Julien Sahli)