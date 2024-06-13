begin by rendering the corresponding psf :
python ../../../../scripts/simulator/generate_psf.py --input_fp=psf.png --output_fp=psf.npy --mode=mask --camera_fp=demo-cam.npy

then render the scene:
python ../../../../scripts/simulator/render_scene.py --radiance_fp=scene.png --depths_fp=scene_dep.png --psf_fp=psf.npy --out_fp=out.png --camera_fp=demo-cam.npy --scene_fp=demo-scene.npy
