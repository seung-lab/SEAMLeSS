#!/usr/bin/env bash
# sample script to setup a virtual environment able to run SEAMLeSS

sudo apt-get install virtualenv g++ python3-dev

# set up a virtual environment
virtualenv -p python3 ~/venvseamless
source ~/venvseamless/bin/activate
pip install numpy # needed for CloudVolume to install correctly
pip install -r requirements.txt
# NOTE: might need to manually install a different version of PyTorch
# depending on the available CUDA drivers. See https://pytorch.org/get-started
python setup.py develop

# Setting up CloudVolume credentials
# see https://github.com/seung-lab/cloud-volume
mkdir -p ~/.cloudvolume/secrets/
mv google-secret.json ~/.cloudvolume/secrets/ # to copy from existing

# activate tab completion
activate-global-python-argcomplete --user
# if the above doesn't work for tab completion, run the following line
# for each python script instead
# eval "$(register-python-argcomplete ./<script_name>.py)"

# speed up startup time
sudo nvidia-persistenced --persistence-mode
