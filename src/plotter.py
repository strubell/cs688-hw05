'''
CS688 HW05: Restricted Boltzmann Machine for binary classification

Utilities for plotting data

@author: Emma Strubell
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# plot a 10x10 array of 28x28 greyscale images
def plot100(images):
    xplots = 10
    yplots = 10
    fig, ax = plt.subplots(xplots, yplots)
    for i in range(xplots):
        for j in range(yplots):
            ax[i][j].axis('off')
            ax[i][j].imshow(images[i*10+j].reshape((28,28)), interpolation='nearest', cmap='gray')

# plot 1 greyscale image
def plot1(im):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im.reshape((28,28)), interpolation='nearest', cmap='gray')

# plot the given time series (in this case energies over iterations)
def plot_errors(errors):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Test Error Rate vs. Sampling Iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error Rate")
    ax.plot(errors)