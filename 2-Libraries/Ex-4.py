#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def main():

    x, y =np.loadtxt('data4.dat', usecols=(0,1), unpack=True)
    plt.scatter(x,y, marker='o',color='red', )
    plt.grid()
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.title('Charged particles')
    plt.savefig('output.png')
    plt.show()

    



if __name__=='__main__':
    main()
