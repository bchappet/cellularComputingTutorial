import matplotlib.pyplot as plt
import numpy as np


def plotArray(array):
    plt.imshow(np.copy(array),cmap='gray_r')
    
def plotIterations(gof,nbIteration=1):
    gof.set(np.random.random(gof.shape) < 0.5)
    for i in range(nbIteration):
        if i != 0:
            gof.run()
        plt.subplot(100 + nbIteration*10+i+1)
        plotArray(gof.get())
        plt.xticks([])
        plt.yticks([])

