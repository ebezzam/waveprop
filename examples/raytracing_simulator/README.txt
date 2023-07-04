This code is a raytracing PSF simulator that was part of a project on LenslessPiCam:
https://github.com/LCAV/LenslessPiCam

It is provided as is and likely needs some adaptation to be runnable as it depends on the aforementioned project.

Is is also quite unefficient and serves more as a demonstration purpose, especially since the raytracing is done entierely in python, and that most of the methods in simulator/utils can certainly be replaced by their respective equivalents from common librairies such as opencv or similar.

If you look for a more efficient code, please check this other branch of the repository, proposing a solution based on the excellent mitsuba:
https://github.com/Julien-Sahli/waveprop/tree/mitsuba



To give it a try, run:

scripts/simulator/generate_psf.py to generate a psf from a mask / height map / normal map 
scripts/simulator/render_scene.py to generate lensless data from a psf and a scene

use the examples provided in the data folder!

To generate the scene, run scripts/conversion/blender_export.py directly from blender (see instructions in the script)

The lensless data should be reconstructible with LenslessPiCam, but the associated psf need to be padded with black so that its resolution matches the one from the lensless data, because of how LenslessPiCam works. It also may need to be rotated of 180°.



Also check the corresponding medium posts!

https://medium.com/@julien.sahli/simulating-lensless-camera-psfs-with-ray-tracing-a224ca11f758
https://medium.com/@julien.sahli/simulating-lensless-camera-data-from-3d-scenes-a3da3daf50e