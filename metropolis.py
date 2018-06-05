import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
import sys
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from scipy.stats import multivariate_normal

mus = np.array([5, 5])
sigmas = np.array([[1, .9], [.9, 1]])

def circle(x, y):
    return (x-1)**2 + (y-2)**2 - 3**2


def pgauss(x, y):
    return st.multivariate_normal.pdf([x, y], mean=mus, cov=sigmas)

def multivariate_gaussian(pos, mu, Sigma):
    n = mu.shape[0]

    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

def metropolis_hastings(p, iter=1000):
    figPlot = plt.figure(figsize=(10,10))

    axPlot = figPlot.add_subplot(311)
    axGraph = figPlot.add_subplot(312)
    axContour = figPlot.add_subplot(313)

    x, y = 0. , 0.
    k = 0

    samples = np.zeros((iter, 2))
    labels = ['point{0}'.format(i) for i in range(iter)]

    for i in range(iter):
        # Generate acceptance ratio
        x_star, y_star = np.array([x, y]) + np.random.normal(size=2)
       
        # Verify uniform random number is greater than acceptance ratio
        if np.random.rand() < p(x_star, y_star) / p(x, y):
            x, y = x_star, y_star
            samples[k] = np.array([x, y])
            k = k + 1
            
            if(k > 1):
                
                ###### Observed Samples
                axGraph.clear()
                x_mu  = np.mean(samples[0:k,0])
                y_mu  = np.mean(samples[0:k, 1])
                cov = np.cov(samples[0:k].T)
                xg, yg = np.random.multivariate_normal((x_mu, y_mu),cov, 50).T
                axGraph.plot(xg, yg, 'x') # = rv.pdf(rv)


                ###### Normal Dist Random Generated Samples
                axPlot.clear()
                axPlot.plot(samples[0:k, 0], samples[0:k, 1], 'ro', markersize=3, color="red")
                axPlot.plot(samples[0:k, 0], samples[0:k, 1], color="blue")
                for j in range(k):
                    axPlot.annotate( j , (samples[j][0], samples[j][1]))
                axPlot.set_xbound(-2.5, 5)
                axPlot.set_ybound(-2.5, 5)
                axGraph.set_xbound(-2.5, 5)
                axGraph.set_ybound(-2.5, 5)
                axContour.set_xbound(-2.5, 5)
                axContour.set_ybound(-2.5, 5)

                ######## Contour
                axContour.clear()
                N = 60
                X = np.linspace(-2.5, 5, N)
                Y = np.linspace(-2.5, 5, N)
                X, Y = np.meshgrid(X, Y)

                mu = np.array([x_mu, y_mu])
                Sigma = np.array(cov)

                pos = np.empty(X.shape + (2,))
                pos[:, :, 0] = X
                pos[:, :, 1] = Y
                Z = multivariate_gaussian(pos, mu, Sigma)

                cset = axContour.contour(X, Y, Z, zdir='z')
            
            plt.draw()
            plt.waitforbuttonpress()

    return samples




if __name__ == '__main__':

    metropolis_hastings(circle, iter=10000)

    # metropolis_hastings(pgauss, iter=10000)
