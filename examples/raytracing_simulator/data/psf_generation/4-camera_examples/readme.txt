Try different camera settings for generating PSFs.

The list of settings can be displayed with the following :
python ../../../../scripts/simulator/store_settings.py --help

Unspecified settings will be left to default.

Try the following different scenes :
python ../../../../scripts/simulator/store_settings.py --path=default --scene_min_depth=0.1 --scene_max_depth=1
python ../../../../scripts/simulator/store_settings.py --path=close --scene_min_depth=0.2 --scene_max_depth=0.7
python ../../../../scripts/simulator/store_settings.py --path=far --scene_min_depth=2 --scene_max_depth=5
python ../../../../scripts/simulator/store_settings.py --path=small-sensor --sensor_width=0.8
python ../../../../scripts/simulator/store_settings.py --path=big-sensor --sensor_width=5
python ../../../../scripts/simulator/store_settings.py --path=squeezed --diffuser_width=1 --diffuser_height=0.5
python ../../../../scripts/simulator/store_settings.py --path=thick --diffuser_thickness=1

Generate and export the psf as usual, but try the different cameras
python ../../../../scripts/simulator/generate_psf.py --input_fp=heights.png --output_fp=result.npy --mode=height --oversample 0.5 --save_normals --camera_fp default-cam.npy

python ../../../../scripts/conversion/npy_to_tiff.py result.npy

(Height map credits : Julien Sahli)