# SEAMLeSS

Working repository for active development of the SEAMLeSS architecture for the purposes of aligning Basil. This codebase is 
not intended to remain consistent with the 2018 NIPS submission "Siamese Encoding and Alignment by Multiscale Learning with 
Self-Supervision."

## Training

### Key Arguments
To control training, it is best to use arguments/parameters so that they are saved and archived by argparse. To see a list of all of the parameters available in training, run

`python train.py -h`

This will also show you a description for the purpose of each parameter.

The key parameters are the `lambda` family:

`lambda1` controls the weighting of the total smoothness penalty against the similarity penalty (this is the most important/frequently tuned parameter)

`lambda2` controls the relative weight of the smoothness penalty in the region near, but not within defects (cracks and folds); this is usually a value in the range (0.01,0.1)

`lambda3` controls the relative weight of the smoothness penalty in the regions *within* defects; this is almost always 0

`lambda4` is like lambda2, but for the similarity penalty; it is usually a value in the range (5,10), which focuses the net on fixing the defects

`lambda5` is like lambda3, but for the similarity penalty; it is almost always 0

`lambda6` is the coefficient of the consensus penalty used for eliminating drift; it is usually in the range 0.1-10 (this requires more investigation)

Some other key parameters:

`mask_neighborhood_radius` controls the radius in pixels of the neighborhood regions used by `lambda2` and `lambda4`; usually a value in the range (75,150)

`lr` controls the learning rate; usually ~0.0003

`--state_archive` is a filepath to the network archive you'd like to start training from

### Invoking Training

You can begin training or fine-tuning by calling

`python train.py [--param1_name VALUE1 --param2_name VALUE2 ...] EXPERIMENT_NAME`

**If you use the same experiment name twice, you will overwrite the outputs from the older version. It is highly recommended that you use unique experiment names.**

### Fine-tuning
An example invocation of training to fine-tune a network called 'matriarch_na3' would be:

`python train.py --state_archive pt/matriarch_na3.pt --size 8 --lambda1 2 --lambda2 0.04 --lambda3 0 --lambda4 5 --lambda5 0 --mask_smooth_radius 75 --mask_neighborhood_radius 75 --lr 0.0003 --trunc 0 --fine_tuning --hm --padding 0 --vis_interval 5 --lambda6 1 fine_tune_example`

### Training from scratch (TODO)


## Generating Data

Run `gen_stack.py` as `python gen_stack.py --count NUMBER_OF_SAMPLES --source CLOUD_VOLUME_PATH DATASET_NAME`, for example 
`python gen_stack.py --count 100 --source basil_v0/raw_image basil_v0`. This creates a training dataset of shape (100, 50, 
1152, 1152) at mip level 5, that is, 100 stacks of 50 slices of size 1152 x 1152. To change the number of slices per stack, 
image size, or mip level, pass the `--stack_height`, `--dim`, or `--mip` arguments. To generate the corresponding test set, 
pass the `--test` flag; samples are partitioned such that train samples come from (1 <= z < 800) and test samples come from 
(800 <= z < 1000).

## Network Histories
Using the net_hist tool (just type `nh` into the command line if you have my command line tools; `net_hist.py` if you don't), you can see the history of training for a particular network.

## Conventions and Quirks

During its lifetime, SEAMLeSS has developed (and hopefully generally adhered to) many conventiones and likewise accumulated many quirks, some due to PyTorch's design and some more arbitrary. Here are some things to keep in mind to avoid *losing* your mind:

* **Know what you're working with.** PyTorch and NumPy can interact strangely (and generally don't mix), so if you're stuck with some weird behavior, a good debugging step can be to check the types of the objects you're working with. For example, some NumPy functions will hang for a **very** long time (minutes+) without crashing; if you make a seemingly innocuous change to your code and all of a sudden it hangs without crashing, check your types. Some of the helper functions are type-agnostic (`save_chunk` adapts to either PyTorch Variables or Tensors or NumPy arrays), but in general it's good to actually know what you're dealing with.

Convert from PyTorch to NumPy with `v.data.numpy()` if `v` is a `Variable`, or `v.numpy()` if `v` is a `Tensor`. If the `Variable` or `Tensor` is on the GPU, you need to insert a call to `.cpu()` before `.numpy()`, so: `v.data.cpu().numpy()`. Convert from NumPy to PyTorch with `torch.from_numpy(v)` or `torch.FloatTensor(v)` if `v` is an `ndarray`. **Be careful: `torch.from_numpy(v)` will infer the type of the new `Tensor` from `v`, and the newly-created `Tensor` will share memory with `v`. *This means changes to `v` will be reflected in the new `Tensor`; it is generally safer to use `torch.Tensor(v)` (which returns the default `Tensor` type) or `torch.FloatTensor` (or something else from the `Tensor` family to use an explicit type).**

* **Size matters.** There are two conventions for working with sizes in this project. In some areas of the code, we work with images as 2D arrays, in others we work with PyTorch convolution sizes, which means a single 2D image will have the shape `(1,1,dim,dim)` if it has an edge length `dim`. This is a matter of convenience, where sometimes it is useful not to have to constantly ignore the first two axes of a 2D image; however, it can also be a source of confusion and annoyance. If you're getting dimension mismatch errors or similar, check the sizes (`v.size()` for a PyTorch `Variable` or `Tensor`, `v.shape` for a NumPy `ndarray`).

* **PyTorch is unfinished** (as of July 2018). This means that there are bugs, and sometimes stuff won't work and it's not your fault. [Here is an example](https://github.com/pytorch/pytorch/issues/7258). Admittedly this is rare, and PyTorch is approaching production stability, but keep this in mind.

* **Masks make it all work.** SEAMLeSS has come to rely heavily on masks in order to interpret and penalize predictions correctly. There is an important distinction between **masks** and **weights**. **Masks** are interpreted using the `lambda` family to generate **weights.** Masks are abstract images that contain a class label (integer) for each pixel in an image; weights are (generally 2D) scalar fields that represent an intended re-weighting of, for example, the error contribution of each pixel ((i,j) location) in an image or vector field. Masks currently follow a **ternary** convention: for a given pixel, a 0 means that the pixel is a 'normal' pixel (far from a defect); a 1 means a pixel near, but not immediately on top of, a defect; a 2 means a pixel that is within a defect.

* **`lambda` is everything (not really, but kind of).** Any time you change the logic of masks and weighting or any of the components of the loss functions, **you will likely need to re-tune the `lambda` parameters**, in most cases just `lambda1'. Sometimes a change to masking logic or updates to loss functions appear to break SEAMLeSS when really they just changed the range of values of a particular contribution to the loss function. Tread carefully.

* **There are tricky local minima out there.** The most common is when the network collapses to a state when it always outputs the zero field. **This is almost always an unrecoverable, pathological solution; if you get to this point, you probably want to quit that training/fine-tuning session and re-think or re-tune.** A notable exception to this is very early in training a **new** network; it is common for the top layer (and only the top layer) to sit in this solution of the zero field for a while (quarter or half of an epoch) before learning something useful.

* **Loss function quirks.**

## TODOs

There are some unresolved mysteries and partially-completed research endeavors in SEAMLeSS as it stands:

* What is the best smoothness penalty? We have several implemented, but others exist too.

* What is the best similarity penalty?

* How do we normalize inputs in order to make the error contributions consistently interpretable? The MSE between two images represented as floating point values in [0,1] and the exact same images represented as unsigned integers in [0,255] will be much different.