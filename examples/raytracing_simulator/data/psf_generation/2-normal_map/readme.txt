Generate a new PSF using a normal map :
python ../../../../scripts/simulator/generate_psf.py --input_fp=normals.jpg --output_fp=result.npy --mode=normals

Export the PSF to tiff images to display them :
python ../../../../scripts/conversion/npy_to_tiff.py result.npy

Oversample the normal map to produce a greater output quality :
python ../../../../scripts/simulator/generate_psf.py --input_fp=normals.jpg --output_fp=result_highres.npy --mode=normals --oversample=2

And export it
python ../../../../scripts/conversion/npy_to_tiff.py result2.npy

(Normal map credits : https://www.cadhatch.com/seamless-water-textures)