#!/usr/bin/env python
import numpy as np
import pandas as pd

def main():

    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight','Acceleration', 'Model Year', 'Origin']

    raw_dataset = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)

    means=raw_dataset.mean()
    print('Means:')
    print(means)

    res=raw_dataset[raw_dataset['Cylinders']==3]
    print('Filtered items with 3 cylinders')
    print(res)



if __name__=='__main__':
    main()
