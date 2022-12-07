#!/usr/bin/env python
import numba as nb
import numpy as np
import time


def basic_dot(A,v):
    N=v.size
    ans=np.zeros(N)
    
    for i in range(N):
        for j in range(N):
            ans[i]+=A[i][j]*v[j]

    return ans

@nb.njit(parallel=True)
def numba_dot(A,v):
    N=v.size
    ans=np.zeros(N)
    
    for i in range(N):
        for j in range(N):
            ans[i]+=A[i][j]*v[j]

    return ans


def main():
    N = int(input('Enter N: '))
    v = np.random.rand(N).astype(np.float64)
    A = np.random.rand(N, N).astype(np.float64)
    start_t=time.time()
    res=basic_dot(A,v)
    print(f'Execution time, basic_dot: {time.time()-start_t}')

    start_t=time.time()
    res=A.dot(v)
    print(f'Execution time, numpy dot: {time.time()-start_t}')

    start_t=time.time()
    res=numba_dot(A,v)
    print(f'Execution time, numba dot (first run): {time.time()-start_t}')

    start_t=time.time()
    res=numba_dot(A,v)
    print(f'Execution time, numba dot (second run): {time.time()-start_t}')





    
    
if __name__=='__main__':
    main()
