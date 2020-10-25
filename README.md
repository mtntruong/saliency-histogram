# saliency-histogram  
C++ code for our paper:  
Single object tracking using particle filter framework and saliency-based weighted color histogram  
M.T.N. Truong, M. Pak, S. Kim  
Multimedia Tools and Applications, vol. 77, no. 22, pp. 30067-30088, 2018  
https://dx.doi.org/10.1007/s11042-018-6180-5

# Usage
First install OpenCV 3.0+ then compile the program by executing `make`, then run the program using this syntax

    ./particle_tracker -o <output filename> -p <number of particles> <path/to/video>
For example

    ./particle_tracker -o ~/output.avi -p 300 ~/basketball.avi
After executing the program, a window will appear showing the first frame of the video, use mouse pointer to locate the target (click and drag to draw a bounding box around the target). Once the mouse button is released, the tracking process starts immediately.

# Further information
The particle filter framework is originally implemented by Kevin Schluff (https://bitbucket.org/kschluff/particle_tracker). We integrated the saliency-based weighted color histogram into this implementation.
