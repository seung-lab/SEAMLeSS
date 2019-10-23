# Generating training data  
Training data is an H5 file, with a dataset name 'main', which stores
a 4D array with (batch)xZxHxW shape.

To generate a set of random cutouts from a CloudVolume, use`gen_stack.py`,  
To convert a CloudVolume cutout into an H5, use `make_cutout.py`.
