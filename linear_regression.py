import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt


# y = w x + e


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
    for i in range(n):
       
    
        x_samples[i] = np.random.uniform(-10, 10)
        w_samples[i] = np.random.normal(mu, sigma)
        y_samples[i] = x_samples[i] * w_samples[i]
        k = k + 1

        ###### W Samples vs. Sample Number
        axGraph.clear()
        w_mu  = np.mean(w_samples[0:k])
        cov = np.var(w_samples[0:k])
        axGraph.plot(w_samples[0:k], 'x') 

        ###### Updated linear function y = w * x + e
        t1 = [-5, 5]
        t2 = [-5 * w_mu, 5 * w_mu]
        axPlot.plot(t1, t2) 

        # ######## Contour
        # axContour.clear()
        # N = 60
        # X = np.linspace(-2.5, 5, N)
        # Y = np.linspace(-2.5, 5, N)
        # X, Y = np.meshgrid(X, Y)

        # mu = np.array([x_mu, y_mu])
        # Sigma = np.array(cov)

        # pos = np.empty(X.shape + (2,))
        # pos[:, :, 0] = X
        # pos[:, :, 1] = Y
        # Z = multivariate_gaussian(pos, mu, Sigma)

        # cset = axContour.contour(X, Y, Z, zdir='z')
        
        plt.draw()
        plt.waitforbuttonpress()

    return samples






if __name__ == '__main__':
    linear_regression()