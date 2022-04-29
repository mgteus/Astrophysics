import os
import pandas as pd
import csv
import numpy as np

def find_missing_values(df:pd.DataFrame = pd.DataFrame()) -> dict:
    """
    Function which receives a dataframe and returna a dict with the index of
    rows with MISSING values and their column

    example: {'index':[i, j], 'cols':[col1, col2]}
    """

    aux_dict = {'index': [], 
                'cols': []}

    if isinstance(df, pd.DataFrame) and df.shape[0] > 0:
        for col in df.columns:
            for i in range(df.shape[0]):
                val = df[col].iloc[i]
                if val == "MISSING":
                    aux_dict['index'].append(i)
                    aux_dict['cols'].append(col)
        return aux_dict

    else:
        return aux_dict

def read_dat_file(path:str = '', cols:list=[]) -> pd.DataFrame:
    """
    Function to read a .dat file returning a pandas dataframe
    """
    df= pd.DataFrame()
    if os.path.exists(path) and len(cols) > 0:
        aux_dict = {col:[] for col in cols}
        with open(path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            if line.startswith("#"):
                continue

            vals = line.replace("\n", '').replace('\t', '').replace('****', 'MISSING').replace(" ", '').split('&')
            for col, val in zip(aux_dict.keys(), vals):
                aux_dict[col].append(val)
        df = pd.DataFrame.from_dict(aux_dict)
        return df
    else:
        return df

def fill_missing_values(df:pd.DataFrame = pd.DataFrame(), aux_dict: dict = {}) -> pd.DataFrame:
    """
    Function to complete the missing values on dataframe based on which column
    has a missing value
    """
    aux_dict_2 = aux_dict.copy()
    aux_dict_2['values'] = []       # new values

    for index, col in zip(aux_dict['index'], aux_dict['cols']):
        aux_values = dict(df.iloc[index])
        
        if col == 'X': # missing X value
            L  = float(aux_values['L'])
            B  = float(aux_values['B'])
            Rs = float(aux_values['R_Sun'])
            X  = Rs*np.cos(L)*np.cos(B) # calculated value for X
            aux_dict_2['values'].append(round(X, 4))

        elif col == 'R_Sun': # missing R_Sun value
            L  = float(aux_values['L'])
            B  = float(aux_values['B'])
            X  = float(aux_values['X'])
            Rs = X / (np.cos(B)*np.cos(L)) # calculated value for Rs
            aux_dict_2['values'].append(round(Rs,4))

        elif col == 'Z':
            B  = float(aux_values['B'])
            Rs = float(aux_values['R_Sun'])
            Z  = Rs*np.sin(B)
            aux_dict_2['values'].append(round(Z, 4))

        elif col == 'R_gc': # missing R_cg value
            X   = float(aux_values['X'])
            Y   = float(aux_values['Y'])
            Z   = float(aux_values['Z'])
            L  = float(aux_values['L'])
            B  = float(aux_values['B'])

            R_s = np.sqrt(X**2 + Y**2 + Z**2)
            R_0 = 8.0

            R_cg = np.sqrt(R_s**2 + R_0**2 - 2*R_0*R_s*np.cos(B)*np.cos(L))
            aux_dict_2['values'].append(round(R_cg, 4))
        else: # one unexpected value from a unexpected column is missing
            aux_dict_2['values'].append("NAN")

    return aux_dict_2

def set_values_on_df(df:pd.DataFrame = pd.DataFrame(), aux_dict:dict = {}) -> pd.DataFrame:
    """
    Function which set new values on specific locations on a dataframe based on
    a dict with the information
    """
    for col, index, val in zip(aux_dict['cols'], aux_dict['index'], aux_dict['values']):
        df[col].iloc[index] = val
    
    return df



if __name__ == '__main__':
    df = read_dat_file('Relatorio1\data.dat', ['ID-Name', 'RA (2000)',
                                         'DEC', 'L', 'B','R_Sun',
                                         'R_gc', 'X', 'Y', 'Z'])
    d = find_missing_values(df)

    x = fill_missing_values(df, d)

    df = set_values_on_df(df, x)

    df.to_csv("Relatorio1\data_completo.csv", index=False)
    