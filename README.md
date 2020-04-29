# VideoProcess
Tools to colorize and upscale resolution for images and videos and add FPS to videos. All that thanks to Machine Learning.
### Tools :
```
White Balance
Denoise
Video Stabilization
DAIN (extern tool)
Super Resolution
Colorization
```
# White Balance
Color White Balance

from opencv xphoto contrib module 

# Denoise
Performs image denoising using the Block-Matching and 3D-filtering algorithm

from opencv xphoto contrib module

# Video Stabilization
Perform video stabilization.

From opencv videostab contrib module. 

# DAIN

"Dain-App is a free app that let you take any form of media like movies, stop-motion, anime, cartoons,
sprites, etc and interpolate new frames, generating a bigger frame-rate from the original file."

The link : https://grisk.itch.io/dain-app

See tutos to use it. I won't explain.

# Super Resolution
This module contains several learning-based algorithms for upscaling an image.
### The models :
1. EDSR
    - x2, x3, x4 trained models available
    - Advantage: Highly accurate
    - Disadvantage: Slow and large filesize

2. ESPCN
    - x2, x3, x4 trained models available
    - Advantage: It is tiny and fast, and still performs well.
    - Disadvantage: Perform worse visually than newer, more robust models.
    
3. FSRCNN
    - Advantage: Fast, small and accurate
    - Disadvantage: Not state-of-the-art accuracy
    
4. LapSRN
    - x2, x4, x8 trained models available
    - Advantage: The model can do multi-scale super-resolution with one forward pass.
    - Disadvantage: It is slower than ESPCN and FSRCNN, and the accuracy is worse than EDSR.
    
from opencv dnn superres contrib module.
I hacked a part of source to include in mine to use possibility of cuda dnn.

# Colorization
Colorize Black and white (even colored) images and videos.
It uses machine learning model to predict colors.

# How it works
Windows Only. Tested on Win 10.
All the modules, except DAIN app, uses opencv 4.2 cuda 10.2.
So it's better to have a Nvidia graphic card to perform max.
Some of the tools are quite slow, even on GPU. Using them on CPU should take a lot of time.

For the Windows users, i share the dll's from opencv :
Opencv downloads : (http://docs.encima.fr/index.php/s/zJnnmqgzd949cJD)

These files should be near .exe files or on a place that is on windows path in env vars.




