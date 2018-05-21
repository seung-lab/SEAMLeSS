#####################################################################################
# analysis_helpers.py
#
# Functions for looking at images and vector fields,
# functions for managing all of my recorded runs.
#####################################################################################

import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
img_w = 128

# Run management
runs_dir = "runs/"

def build_path(seed, mom, lr_init, lr_decay):
    return runs_dir + \
           str(seed) + "_seed/" + \
           "{0:.4f}".format(mom) + "_mom/" + \
           "{0:.3e}".format(lr_init)  + "_lr/"  + \
           "{0:.4f}".format(lr_decay) + "_gam/"

def get_perf(seed, mom, lr_init, lr_decay):
    curve_file = build_path(seed, mom, lr_init, lr_decay) + "perf.txt"
    f = open(curve_file, 'r')
    performance = 1
    for l in f.readlines():
        arr = l.strip().split()
        if arr[0] == "Epoch":
            performance = float(arr[-1].split("=")[1])
    f.close()
    return performance

def get_inventory(seed=17724, mom=0.9):
    seed_dir = str(seed)+"_seed"
    mom_dir  = "{0:.4f}".format(mom) + "_mom"
    # learning rates
    lrs = {}
    for lr_dir in os.listdir(runs_dir + seed_dir+"/" + \
                             mom_dir+"/"):
        lr = lr_dir[:len(lr_dir)-3]
        # learning_decays
        gammas = []
        for gam_dir in os.listdir(runs_dir + seed_dir+"/" + \
                                  mom_dir+"/" + lr_dir+"/"):
            gam = gam_dir[:len(gam_dir)-4]
            gammas.append(float(gam))
            lrs[float(lr)] = sorted(gammas)
    return lrs

def show_inventory(seed=17724, mom=0.9):
    #print "seed = "+str(seed)+", ", "momentum = {0:.4f}".format(mom)
    lrs = get_inventory(seed, mom)
    for lr in lrs:
        #print "{0:.3e}".format(lr), ": ",
        for gamma in lrs[lr]:
            pass
            #print "{0:.4f}".format(gamma),
        #print ""

def is_complete(seed, mom, lr, gam):
    with open(build_path(seed, mom, lr, gam)+"perf.txt") as f:
        for i, l in enumerate(f):
            pass
    n_lines = i+1
    if n_lines > 490: raise ValueError('File length too long')
    return n_lines == 490

# Visualizing curves and samples

def display_img(a, b, c):
    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(a, cmap='Greys')
    plt.subplot(1,3,2)
    plt.title("Transformed")
    plt.imshow(b, cmap='Greys')
    plt.subplot(1,3,3)
    plt.title("Predicted")
    plt.imshow(c, cmap='Greys')
    plt.show()

def display_v(V_pred, name):
    X, Y = np.meshgrid(np.arange(-1, 1, 2.0/V_pred.shape[-2]), np.arange(-1, 1, 2.0/V_pred.shape[-2]))
    U, V = np.squeeze(np.vsplit(np.swapaxes(V_pred,0,-1),2))
    colors = np.arctan2(U,V)   # true angle
    plt.title('V_pred')
    plt.gca().invert_yaxis()
    Q = plt.quiver(X, Y, U, V, colors, scale=6, width=0.002, angles='uv', pivot='tail')
    qk = plt.quiverkey(Q, 10.0, 10.0, 2, r'$2 \frac{m}{s}$', labelpos='E', \
                       coordinates='figure')

    plt.savefig(name + '.png')
    plt.clf()

def vis_grid(sample_file):
    f = h5py.File(sample_file, 'r')

    for i in range(9):
        img_orig = f["img_orig"][i,:,:]
        img_tran = f["img_tran"][i,:,:]
        img_pred = f["img_pred"][i,:,:]
        v_pred   = f["v_pred"][i,:,:,:]

        plt.subplot(3,3,i+1)

        X, Y = np.meshgrid(np.arange(-1, 1, 2.0/img_w), np.arange(-1, 1, 2.0/img_w))
        U, V = np.squeeze(np.vsplit(np.swapaxes(v_pred,0,-1),2))
        colors = np.arctan2(U,V)   # true angle
        plt.gca().invert_yaxis()
        Q = plt.quiver(X, Y, U, V, colors, scale=6, width=0.002, angles='uv', pivot='tail')
        qk = plt.quiverkey(Q, 10.0, 10.0, 2, r'$2 \frac{m}{s}$', labelpos='E', \
                           coordinates='figure')

    plt.show()
    f.close()

def vis_sample(sample_file, i=0):
    f = h5py.File(sample_file, 'r')
    img_orig = f["img_orig"][i,:,:]
    img_tran = f["img_tran"][i,:,:]
    img_pred = f["img_pred"][i,:,:]
    v_pred   = f["v_pred"][i,:,:,:]
    display_img(img_orig, img_tran, img_pred)
    display_v(v_pred)
    f.close()

def vis_curve(curve_file, showTest=False):
    f = open(curve_file, 'r')

    train_iters = []; train_losses = []
    test_iters = []; test_losses = []

    isTrain = True
    for l in f.readlines():
        arr = l.strip().split()
        if arr[0] == 'Test...': isTrain = False
        if len(arr) != 7: continue
        if isTrain:
            train_iters.append(float(arr[3][:-1]))
            train_losses.append(float(arr[6].split('=')[-1]))
        else:
            test_iters.append(float(arr[3][:-1]))
            test_losses.append(float(arr[6].split('=')[-1]))

    plt.plot(train_iters, train_losses, c='b')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training-- end: ' + str(train_losses[-1]))
    plt.grid(True)
    plt.show()

    if showTest:
        plt.plot(test_iters, test_losses, c='b')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Testing-- end: ' + str(test_losses[-1]))
        plt.grid(True)
        plt.show()

    f.close()
