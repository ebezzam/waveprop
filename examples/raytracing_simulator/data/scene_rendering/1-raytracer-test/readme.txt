A scene that is small enough to be rendered with the raytracer. We will compare it to the convolution version after that.

python ../../../../scripts/simulator/render_scene.py --mode=raytracing --radiance_fp=scene.png --depths_fp=scene_dep.png --psf_fp=psf.png --out_fp=out.png --dest_shape 42 42 --camera_fp good-cam.npy --scene_fp good-scene.npy
