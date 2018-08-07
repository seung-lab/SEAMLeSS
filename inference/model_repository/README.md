# Models here

Only those models that have been tested and ready to go for large scale inference should be uploaded to Git.

Models are named according to the following convention:

VERSION_LAYERS_SKIPS

and use the .pt extension. For example, if the version 1 pyramid has 6 layers and only uses the
top 3 residuals, it would be named 1_6_3.pt. If the second version of the pyramid has 7 layers and only uses
the top 3 residuals, it would be named 2_7_4.pt. If the third version of the pyramid has 5 layers and uses all
of the residuals, it would be named 3_5_0.pt.
