*** This is a medium version of the Smartphone Image Denoising Dataset (SIDD) 
[sRGB images]
=============================================================================

The "Data" directory contains 320 image pairs (noisy and ground-truth, in sRGB space) representing 160 scene instances, two images from each scene instance. 

These images can be used for training/learning purposes.

For each image, the following is provided in one directory:

- Two noisy sRGB image (.PNG).
Gamma corrected, without any tone mapping.

- Two ground truth sRGB image (.PNG), gamma corrected, without any tone mapping.
Gamma corrected, without any tone mapping.

So, the total number of image files in the "Data" directory is 160 x (2 + 2) = 640.


*** Image directory naming convention:
======================================

Each image is stored in a directory with the name of the scene instance as follows:

<scene-instance-number>_<scene_number>_<smartphone-code>_<ISO-level>_<shutter-speed>_<illuminant-temperature>_<illuminant-brightness-code>

where:

"smartphone-code" is one of the following:

GP: Google Pixel
IP: iPhone 7
S6: Samsung Galaxy S6 Edge
N6: Motorola Nexus 6
G4: LG G4

and 

"illuminant-brightness-code" is one of the following:

L: low light
N: normal brightness
H: high exposure

For example, the following directory name:

0052_002_S6_01600_01000_5500_N

means that this is scene instance 52, from scene 2, captured be Samsung Galaxy S6 Edge (S6), using ISO level of 1600, using shutter speed of 1000 (i.e., exposure time of 1/1000 second), under illuminant temperature of 5500K, under normal (N) brightness.


*** Scene_Instance.txt
======================

This file contains the names of all directory names inside the "Data" directory, which also represent the scene instance names.