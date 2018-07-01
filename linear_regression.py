import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# y = w x + e

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def linear_regression():
    figPlot = plt.figure(figsize=(10,10))

    axPlot = figPlot.add_subplot(311)
    axGraph = figPlot.add_subplot(312)
    axContour = figPlot.add_subplot(313)

    mu = 16
    sigma = 10.8
    n = 1000

    k = 0

    labels = ['point{0}'.format(i) for i in range(n)]

    x_samples = np.zeros(1000)
    w_samples = np.zeros(1000)
    y_samples = np.zeros(1000)
    w_mu_samples = np.zeros(1000)
    cov_samples = np.zeros(1000)
    gaussian_plots = np.zeros(1000)
    for i in range(n):
       
    
        x_samples[i] = np.random.uniform(-10, 10)
        w_samples[i] = np.random.normal(mu, sigma)
        y_samples[i] = x_samples[i] * w_samples[i]
        w_mu  = np.mean(w_samples[0:k])
        cov = np.var(w_samples[0:k])

        w_mu_samples[i] = np.mean(w_samples[0:k])
        cov_samples[i] = cov
        k = k + 1

        ###### W Samples vs. Sample Number
        axGraph.clear()
        axGraph.plot(w_samples[0:k], 'x') 

        ###### Updated linear function y = w * x + e
        t1 = [-5, 5]
        t2 = [-5 * w_mu, 5 * w_mu]
        axPlot.plot(t1, t2) 

        ###### Updated W Normal Distribution
        axContour.clear()
        
        for j in range (k):
            gaussian_plot = np.linspace(w_mu_samples[i] - 3*cov_samples[i], w_mu_samples[i] + 3*cov_samples[i], 100)
            axContour.plot( gaussian_plot, mlab.normpdf(gaussian_plot, w_mu_samples[j], cov_samples[j]))
        
        plt.draw()
        plt.waitforbuttonpress()

    return samples






if __name__ == '__main__':
    linear_regression()