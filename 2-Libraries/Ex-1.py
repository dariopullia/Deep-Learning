#!/usr/bin/env python
import numpy as np


def main():
    a=np.array([[0.5, -1],[-1, 2]], dtype=np.float32)

    print(f'Shape: {a.shape}, Dimensions: {a.ndim}')

    b=a.copy().flatten()

    for i in range(b.size):
        if(i%2==0):
            b[i]=0
    print(b)

if __name__=='__main__':
    main()
