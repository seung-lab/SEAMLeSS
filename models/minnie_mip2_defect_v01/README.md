Trained as follows:
* single sample, no augmentation, until perfect
* add rotation and flip augmentation, train until perfect
* increase to 4 samples, train until loss stops dropping
* increase to 16 samples, again until no more drop
* train on the full dataset (904 samples, all of which include defects)
