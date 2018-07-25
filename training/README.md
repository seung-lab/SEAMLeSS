# SEAMLeSS

** This documentation is a work in progress. Please let me know if you have any questions/suggestions. **

Working repository for active development of the SEAMLeSS architecture for the purposes of aligning Basil. This codebase is 
not intended to remain consistent with the 2018 NIPS submission "Siamese Encoding and Alignment by Multiscale Learning with 
Self-Supervision."

[Skip straight to the [General Overview](README.md#general-overview)]

## What's What

* `train.py` : This is where the magic happens. The main logic of taking raw inputs, generating masks and weights, running the net, and computing gradients from the results is here. The main training loop calls `run_pair()`, which calls `run_sample` (or `run_supervised`) once for the input in its normal orientation and once with the input rotated 180 degrees (for computation of consensus penalty, if desired).

* `aug.py` : Augmentation code is here, along with some augmentation-specific convenience functions for rotating things, specialized random sampling, and mask generation. Key players are `aug_stacks`, which performs what we refer to as **non-invertible** augmentation to a sequence of stacks of shape (1,H,D,D). That is, we don't 'undo' this augmentation when we compute the loss. The augmentation will be consistent across stacks, except that the first item in the sequence will be 'cut' randomly (to simulate inconsistent edges). This method is used to jitter a stack of EM images along with its masks. The jitter includes translation, slight rotation, and slight scaling. This process allows us to control the size of the displacements that net is trained on. On the other hand, `aug_input` performs **invertible** augmentation to a single slice (*not a stack*). This augmentation includes missing data cutouts, brightness cutouts, tiling and periodic contrasting augmentation, and general brightness augmentation. 

* `helpers.py` : A conglomeration of various general tools that are used across the project. These include wrapper function for saving images or gifs, our custom archive loader, and more.

* `loss.py` : Our loss functions and corresponding wrapper functions. For example, calling loss.smoothness_penalty('jacob') will return a smoothness penalty function that computes the smoothness using the approximate discrete centered Jacobian of the vector field. Supported loss functions are 'lap', 'jacob', 'cjacob', and 'tv'. 'lap' uses the deviation of the vector from the average of its four neighbors as its penalty contribution, and 'tv' uses the [total variation](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7570266) of the field, which is basically the jacobian but using the absolute size of the difference vectors rather than the square.

* `stack_dataset.py` : A wrapper around the PyTorch Dataset class that provides some small extra functionality, namely loading from h5 archives.

* `cv_sampler.py` : Some wrappers around CloudVolume. Is used only by `gen_stack` to generate datasets.

* `gen_stack.py` : Provides functionality for generating datasets by sampling from CloudVolume datasets. Our datasets are h5 binary data archives. Run `python gen_stack.py -h` to see the parameters that the script accepts. Basically, specify a bounding box within a NG path, and you'll get a dataset.

* `combine_h5.py` : So you've run `gen_stack` on several different volumes, but you want to train on all of that data at once. Because `stack_dataset` supports h5 archives with multiple datasets in them, we can run `python combine_h5.py NEW_COMBINED_DATASET_NAME SOURCE_DATASET1.h5 SOURCE_DATASET2.h5 SOURCE_DATASET3.h5 ...`. This will generate NEW_COMBINED_DATASET_NAME.h5, which contains a dataset for each of the inputs, wrapped into one file (**fair warning: because you have to load each dataset in order to combine it, this can take several minutes to run; stand up and use your muscles or something**)

* `requirements.txt` : The dependencies of the project. We recommend using a virtualenv (with Python 2) and installing the dependencies as `pip install -r requirements.txt`.

## Training

### General Overview

**If you only read one section, read this one.**

Training is essentially a few nested loops. Our outer-most loop iterates is the epoch counter, which is simply the number of passes we have made over our training data. Within each epoch we loop over the samples in our dataset. Each sample is a PyTorch Tensor of shape (1,H,D,D), where H is the z dimension size. For each sample, we iterate over this z dimension, training on each adjacent pair of slices.

After loading each training sample (stack of shape (1,H,D,D)) but before we run the network, we do some pre-processing. This pre-processing includes running the defect detection network to generate defect masks, running the normalizer to give standardized contrasting, and performing some 'jittering' on the stack to make the network's task a bit more difficult using the augmentation packages's `aug_stacks` method. The amount of jitter is chosen to be 2^(height-1), using the assumption that each layer of the network corrects roughly one pixel displacements. This jitter is non-invertible; that is, once we apply it, we throw away the original stack, and proceed as if the jittered stack were the raw data. Once this pre-processing is complete, we perform training on each pair of adjacent slices. Each pair (`src`, `target`) receives its own, independent augmentation and is fed to the network. Importantly, **the original slices are retained** for loss computation (the nomenclature in the code is `src` and `target` for the original slices and `input_src` and `input_target` for the augmented versions of the slices that are the actual inputs to the network).

After the network makes a prediction, before we can perform the backward pass, we have to compute our pixel-wise weightings for the similarity and smoothness penalties. This involves interpreting the masks produced by our defect detection framework and converting them to useful weights. Discussion of the conventions of masks is included later, in the [conventions section](README.md#conventions-and-quirks). Once the weights are computed, the error fields (pixel-wise contributions to each error term) are reweighted, and we perform the backwards pass.

In training, we perform this prediction and weighting twice for each pair of adjacent slices in the sample: once in 'normal' orientation, once in 'flipped' orientation (rotated 180 degrees). We keep the outputs (and gradients) around between both samples so that we can compute a consensus penalty between the two predictions. The idea behind this penalty is to encourage consistency of behavior regardless of the orientation of the inputs.

We then do all of this again, and again, and again. There are some details left out here about how we sequentially train the different levels of the pyramid. We'll include that later.

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

`python train.py --state_archive pt/SOME_ARCHIVE.pt --size 8 --lambda1 2 --lambda2 0.04 --lambda3 0 --lambda4 5 --lambda5 0 --mask_smooth_radius 75 --mask_neighborhood_radius 75 --lr 0.0003 --trunc 0 --fine_tuning --hm --padding 0 --vis_interval 5 --lambda6 1 fine_tune_example`

The `fine_tuning` flag essentially reduces the learning rate, trains all parameters together, and trains at the full resolution of the network.

### Training from scratch

The key to training from scratch is identifying when the net is stuck in a minimium that it won't get out from. By far the most common minimum that rears its ugly head is the solution where the net simply outputs a vector field of all zeros. **This most commonly happens when the smoothness penalty is too large relative to the similarity penalty *OR* the learning rate is too large.** The network is particularly sensitive to this phenomenon early in training, because when training the top several layers of the network, the target image that we are matching too is extremely heavily downsampled. This downsampling reduces the dynamic range of the prediction and target images, which has the effect of increasing the smoothness penalty (because the MSE of a misalignment gets smaller if the values in the image are compressed). We've recently implemented a mitigation to this phenomenon by rescaling the images after downsampling to ensure they have the same dynamic range as the inputs, but it hasn't been tested thoroughly.

## Generating Data

We have a hell of a lot of data. We can't train on all of it. Instead, we generate a dataset that is a random (hopefully representative) subsample of the tissue of interest.

Datasets can be generated using `gen_stack.py`. The output will be an A dataset has the shape (N,H,D,D). A single 'sample' is a stack of H consecutive sub-slices from some dataset. N is the number of samples, H is the height/number of slices per sample, D is the side length of each sample (currently we only work with square samples).

**Example:** Run `python gen_stack.py --count NUMBER_OF_SAMPLES --source CLOUD_VOLUME_PATH DATASET_NAME`, for example `python gen_stack.py --count 100 --source basil_v0/raw_image basil_v0`. This creates a training dataset of shape (100, 50, 1152, 1152) at mip level 5, that is, 100 stacks of 50 slices of size 1152 x 1152. 

## Network Histories

Using the net_hist tool (just type `nh` into the command line if you have my command line tools; `net_hist.py` if you don't), you can see the history of training for a particular network. You can change the parameters it shows if you tweak `net_hist.py`.

## Conventions and Quirks

During its lifetime, SEAMLeSS has developed (and hopefully generally adhered to) many conventiones and likewise accumulated many quirks, some due to PyTorch's design and some more arbitrary. Here are some things to keep in mind to avoid *losing* your mind:

* **Know what you're working with.** PyTorch and NumPy can interact strangely (and generally don't mix), so if you're stuck with some weird behavior, a good debugging step can be to check the types of the objects you're working with. For example, some NumPy functions will hang for a **very** long time (minutes+) without crashing; if you make a seemingly innocuous change to your code and all of a sudden it hangs without crashing, check your types. Some of the helper functions are type-agnostic (`save_chunk` adapts to either PyTorch Variables or Tensors or NumPy arrays), but in general it's good to actually know what you're dealing with. Convert from PyTorch to NumPy with `v.data.numpy()` if `v` is a `Variable`, or `v.numpy()` if `v` is a `Tensor`. If the `Variable` or `Tensor` is on the GPU, you need to insert a call to `.cpu()` before `.numpy()`, so: `v.data.cpu().numpy()`. Convert from NumPy to PyTorch with `torch.from_numpy(v)` or `torch.FloatTensor(v)` if `v` is an `ndarray`. **Be careful: `torch.from_numpy(v)` will infer the type of the new `Tensor` from `v`, and the newly-created `Tensor` will share memory with `v`. *This means changes to `v` will be reflected in the new `Tensor`; it is generally safer to use `torch.Tensor(v)` (which returns the default `Tensor` type) or `torch.FloatTensor` (or something else from the `Tensor` family to use an explicit type).**

* **Size matters.** There are two conventions for working with sizes in this project. In some areas of the code, we work with images as 2D arrays, in others we work with PyTorch convolution sizes, which means a single 2D image will have the shape `(1,1,dim,dim)` if it has an edge length `dim`. This is a matter of convenience, where sometimes it is useful not to have to constantly ignore the first two axes of a 2D image; however, it can also be a source of confusion and annoyance. If you're getting dimension mismatch errors or similar, check the sizes (`v.size()` for a PyTorch `Variable` or `Tensor`, `v.shape` for a NumPy `ndarray`).

* **PyTorch is unfinished** (as of July 2018). This means that there are bugs, and sometimes stuff won't work and it's not your fault. [Here is an example](https://github.com/pytorch/pytorch/issues/7258). Admittedly this is rare, and PyTorch is approaching production stability, but keep this in mind.

* **Masks make it all work.** SEAMLeSS has come to rely heavily on masks in order to interpret and penalize predictions correctly. There is an important distinction between **masks** and **weights**. **Masks** are interpreted using the `lambda` family to generate **weights.** Masks are abstract images that contain a class label (integer) for each pixel in an image; weights are (generally 2D) scalar fields that represent an intended re-weighting of, for example, the error contribution of each pixel ((i,j) location) in an image or vector field. Masks currently follow a **ternary** convention: for a given pixel, a 0 means that the pixel is a 'normal' pixel (far from a defect); a 1 means a pixel near, but not immediately on top of, a defect; a 2 means a pixel that is within a defect.

* **`lambda` is everything (not really, but kind of).** Any time you change the logic of masks and weighting or any of the components of the loss functions, **you will likely need to re-tune the `lambda` parameters**, in most cases just `lambda1'. Sometimes a change to masking logic or updates to loss functions appear to break SEAMLeSS when really they just changed the range of values of a particular contribution to the loss function. Tread carefully.

* **There are tricky local minima out there.** The most common is when the network collapses to a state when it always outputs the zero field. **This is almost always an unrecoverable, pathological solution; if you get to this point, you probably want to quit that training/fine-tuning session and re-think or re-tune.** A notable exception to this is very early in training a **new** network; it is common for the top layer (and only the top layer) to sit in this solution of the zero field for a while (quarter or half of an epoch) before learning something useful.

## Open questions

There are some unresolved mysteries and partially-completed research endeavors in SEAMLeSS as it stands:

* What is the best smoothness penalty? We have several implemented, but others exist too.
* What is the best similarity penalty?
* How do we normalize inputs in order to make the error contributions consistently interpretable? The MSE between two images represented as floating point values in [0,1] and the exact same images represented as unsigned integers in [0,255] will be much different.
* How can we best utilize self-supervision as in the original SEAMLeSS paper to quickly learn cracks and folds?