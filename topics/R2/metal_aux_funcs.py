import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_cluster_file(path: str = '') -> pd.DataFrame:
    """
    Function to read clusters file and return a dataframe
    """
    with open(path, 'r') as file:
        lines = file.readlines()

    cols = lines[0].split()[1:]
    #aux_dict = {col:[] for col in cols}
    aux_dict = {}
    for col in cols:
        if col not in aux_dict:
            aux_dict[col] = []
        else:
            aux_dict[col+'_0'] = []
    
    for line in lines[1:]:
        vals = line.split()[:-2]
        for val, col in zip(vals, aux_dict.keys()):
            aux_dict[col].append(float(val))

    df = pd.DataFrame(aux_dict)

    return df


def read_iso_file(path: str='') -> pd.DataFrame:
    """
    Function to read a fixed style table 
    """
    df = pd.DataFrame()
    

    with open(path, 'r') as file:
        lines = file.readlines()
    cols = ['m*(Mo)',  'B' , 'V', 'R',  '(B-V)' ,'(B-R)', '(V-R)']
    
    aux_dict = {col:[] for col in cols}

    if lines[0].startswith('#'):
        lines.pop(0)

    for line in lines:
        vals = line.split()

        for val, col in zip(vals, cols):
            aux_dict[col].append(float(val))


    df = pd.DataFrame(aux_dict)

    return df







if __name__ == '__main__':
    df = read_iso_file(r'topics\R2\clusters\bvr_0.001_1000Myr.dat')
    print(df)
    
