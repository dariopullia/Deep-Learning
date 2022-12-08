#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return -np.sin(x*x)/x + 0.01 * x*x

def main():
    x=np.linspace(-3,3,100)
    y=f(x)

    np.savetxt('output.dat', np.vstack([x,y]))




    plt.plot(x,y, marker='o', label='My function')
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-3,3])
    plt.title('Plotting my function')
    plt.legend()
    plt.savefig('output5.png')
    plt.show()




if __name__=='__main__':
    main()
