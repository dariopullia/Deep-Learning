#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import hyperopt as hp
from tensorflow import keras



tf.random.set_seed(0)


def f(x):
    return 0.05*(x**6 - 2*x**5 - 28*x**4 + 28*x**3 + 12*x**2 - 26*x + 100)




def main():

    x=np.linspace(-5,6,100)
    y=f(x)
    space = hp.hp.uniform('x', -5, 6)

    samples=[hp.pyll.stochastic.sample(space) for _ in range(1000)]
    samples=tf.convert_to_tensor(samples, dtype=tf.float32)

    trials = hp.Trials()
    best = hp.fmin(fn=f, space=space, algo=hp.tpe.suggest, max_evals=2000, trials=trials)

    print(best)
    plt.figure(figsize=(7,10))
    plt.subplot(3,1,1)
    plt.suptitle('Hyperopt TPE')
    plt.title('Function over samples')
    plt.plot(x,y)
    plt.scatter(samples, f(samples), alpha=0.1, label='samples', c='orange')
    plt.scatter(best['x'], f(best['x']), c='g', label='best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.subplot(3,1,2)
    plt.title('X values over trials')
    plt.scatter(range(len(trials.vals['x'])),trials.vals['x'], c='r', label='trials', s=10)
    plt.axhline(y = best['x'], color = 'g', linestyle = '-')
    plt.xlabel('trial')
    plt.ylabel('x')
    plt.legend()

    plt.subplot(3,1,3)
    plt.title('X values histogram')
    plt.hist(trials.vals['x'], bins=50, density=True, label='trials')


    trials = hp.Trials()
    best = hp.fmin(fn=f, space=space, algo=hp.rand.suggest, max_evals=2000, trials=trials)

    print(best)
    plt.figure(figsize=(7,10))
    plt.suptitle('Hyperopt Random')
    plt.subplot(3,1,1)
    plt.title('Function over samples')
    plt.plot(x,y)
    plt.scatter(samples, f(samples), alpha=0.1, label='samples', c='orange')
    plt.scatter(best['x'], f(best['x']), c='g', label='best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.subplot(3,1,2)
    plt.title('X values over trials')
    plt.scatter(range(len(trials.vals['x'])),trials.vals['x'], c='r', label='trials', s=10)
    plt.axhline(y = best['x'], color = 'g', linestyle = '-')
    plt.xlabel('trial')
    plt.ylabel('x')
    plt.legend()

    plt.subplot(3,1,3)
    plt.title('X values histogram')
    plt.hist(trials.vals['x'], bins=50, density=True, label='trials')


    plt.show()


    return 0    





if __name__ == '__main__':  
    main()
