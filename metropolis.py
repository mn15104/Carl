import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton

mus = np.array([5, 5])
sigmas = np.array([[1, .9], [.9, 1]])

def circle(x, y):
    return (x-1)**2 + (y-2)**2 - 3**2


def pgauss(x, y):
    return st.multivariate_normal.pdf([x, y], mean=mus, cov=sigmas)


def metropolis_hastings(p, iter=1000):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x, y = 0. , 0.
    samples = np.zeros((iter, 2))
    k = 0
    labels = ['point{0}'.format(i) for i in range(iter)]
    for i in range(iter):
        # Generate acceptance ratio
        x_star, y_star = np.array([x, y]) + np.random.normal(size=2)
       
        # Verify uniform random number is greater than acceptance ratio
        if np.random.rand() < p(x_star, y_star) / p(x, y):
            x, y = x_star, y_star
            samples[k] = np.array([x, y])
            k = k + 1
            ax.clear()
            ax.plot(samples[0:k, 0], samples[0:k, 1], 'ro', markersize=3, color="red")
            ax.plot(samples[0:k, 0], samples[0:k, 1], color="blue")
            for j in range(k):
                ax.annotate( j , (samples[j][0], samples[j][1]))
            plt.waitforbuttonpress()

    return samples


if __name__ == '__main__':

    metropolis_hastings(circle, iter=10000)

    # metropolis_hastings(pgauss, iter=10000)
