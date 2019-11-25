# Generating training data  
A training set is an H5 file.
The H5 file requires on dataset named `images`, which should be
source and target image pairs. 
The `image` dataset should be stored as a 4D uint8 array (`Kx2xWxH`). 
The first dimension represent `K` samples of images pairs.

The H5 file may optionally contain a second dataset named `masks`.
This dataset should represent a corresponding set of defect masks
(e.g. cracks or folds), which will be used during training.
The `masks` dataset should be stored as a 4D array of the same shape
as the `images` dataset.

<> To generate a set of random cutouts from a CloudVolume, use`gen_stack.py`,  
<> To convert a CloudVolume cutout into an H5, use `make_cutout.py`.
