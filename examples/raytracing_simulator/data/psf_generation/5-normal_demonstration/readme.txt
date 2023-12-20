Demonstrate the light refraction using a normal map.
First, save the following camera settings :
python ../../../../scripts/simulator/store_settings.py --path=settings --scene_min_depth=0.1 --scene_max_depth=1

Generathe the psf :
python ../../../../scripts/simulator/generate_psf.py --input_fp=normals.png --output_fp=result.npy --mode=normals --oversample 0.5 --dest_shape 10 300 300 --camera_fp settings-cam.npy --opengl

Export the PSF to tiff images to display them :
python ../../../../scripts/conversion/npy_to_tiff.py result.npy

When generating the psf, try with and without the opengl flag to parse the normals with either opengl or directx convention and see the difference.
(This normal map use opengl convention, hence using directx will produce a wrong psf, with y-component of the normals inverted)

(Normal map credits : CC BY 4.0 Julian Herzog - https://en.wikipedia.org/wiki/Normal_mapping)