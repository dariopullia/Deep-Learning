#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import  minimize

def true_f(x):
    return np.cos(1.5 * np.pi * x)
    

def main():
    np.random.seed(0)

    x=np.random.rand(30)
    x=np.sort(x)
    y=true_f(x)+np.random.rand(30)*0.1

    x_test=np.linspace(0,1,100)

    def loss(p, func):
        ypred = func(list(p),x)
        return tf.reduce_mean(tf.square(ypred - y)).numpy()

    for degree in [1, 4, 15]:
        res = minimize(loss, np.zeros(degree+1), args=(tf.math.polyval), method='BFGS')
        plt.plot(x_test, np.poly1d(res.x)(x_test), label=f"Poly degree={degree}")

    plt.plot(x_test, true_f(x_test), label="True function")
    plt.scatter(x,y, marker='o', label='data', color='black')
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0,1])
    plt.ylim([-1,1])
    plt.title('Fitting my function')
    plt.legend()
    plt.show()

if __name__=='__main__':
    main()
