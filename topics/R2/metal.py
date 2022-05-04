import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random

from sklearn import cluster
from metal_aux_funcs import read_cluster_file, read_iso_file



def translate_data(V: list = [], BV: list = []) -> list:
    """
    Fuction which tranlate the data to (0,0) point based on the min(values) point
    """
    minBV = min(BV)
    minV  = V[BV.index(min(BV))]

    V_  = [round(v - minV, 4) for v in V]
    BV_ = [round(bv - minBV, 4) for bv in BV]
    return V_, BV_

def create_dict_structure(df:pd.DataFrame = pd.DataFrame(), file: str = "") -> dict:
    """
    Function to create a specific structured dict based on a dataframe
    """
    return {'filename': file, 'V': list(df['V']), 'BV': list(df['(B-V)'])}

def get_iso_files_list(path: str = '') -> list:
    """
    Functio to get a list with the filenames of iso files within the folder 
    """
    files_list = []
    for _, dirs, files in os.walk(path):
        for file in files:
            if file.startswith('bvr_'):
                files_list.append(os.path.join(path, file))


    return files_list

def plot_translated_cluster_data(color:list=[], mag:list=[], file: str = '') -> plt.figure:
    """
    Function to plot a CMD plot
    """
    color = list(color)
    mag  = list(mag)


    fig, ax = plt.subplots(figsize=(16,9), dpi=120)

    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

    plt.title(f'Diagrama CMD para {file} Transladado', fontsize=30)

    plt.scatter(color, mag, label = 'Data', marker='^', alpha=0.1, c='k')
    plt.scatter(min(color), mag[color.index(min(color))], c='r', s=10, label=r'$\alpha$')

    min_bv = min(color)
    min_v  = mag[color.index(min(color))]

    plt.scatter([c - min_bv for c in color], [m - min_v for m in mag], label = 'Data.T', marker='^', alpha=0.2)
    

    ax.invert_yaxis()

    plt.xlabel("B-V", fontsize=15)
    plt.ylabel(" V [mag]", fontsize=15)
    plt.legend(fontsize=20)
    plt.show()


    return

def plot_iso_data_translation(aux_dict: dict = {}) -> plt.figure:
    """
    Function to plot iso data on the same figure
    """
    # dict structure aux_dict = {'file_name':{'V':[], '(B-V)':[]}}

    fig, ax = plt.subplots(figsize=(16,9), dpi=120)

    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

    V  = aux_dict['V']
    BV = aux_dict['BV']

    min_bv = min(BV)
    min_v  = V[BV.index(min(BV))]

    plt.plot(BV, V, label=aux_dict['filename'], alpha=0.2, ls='--', c='k')
    plt.plot([bv-min_bv for bv in BV], [v-min_v for v in V], label=f"{aux_dict['filename']}.T")

    plt.scatter(min(BV), V[BV.index(min(BV))], c='r', label=r'$\alpha$')

    plt.legend()
    ax.invert_yaxis()

    plt.title(f"Diagrama CMD para Isócrona {aux_dict['filename']} Transladado", fontsize=30)
    plt.xlabel("B-V", fontsize=15)
    plt.ylabel(" V [mag]", fontsize=15)
    plt.legend(fontsize=20)
    plt.show()

    return

def plot_iso_and_cluster(cluster:dict = {}, iso:dict = {}) -> plt.figure:
    """
    Function to plot iso and cluster data on the same figure
    """

   # expected arg structure => cluster = {filename:'filename',
   #                                        V  : [],
   #                                        BV : []} 


    fig, ax = plt.subplots(figsize=(16,9), dpi=120)

    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

    cluster['V'], cluster['BV'] = translate_data(cluster['V'], cluster['BV'])

    iso['V'], iso['BV'] = translate_data(iso['V'], iso['BV'])

    plt.plot(iso['BV'], iso['V'], label = iso['filename'], lw = 2, c='r')
    plt.scatter(cluster['BV'], cluster['V'], label = cluster['filename'], alpha=.2)
    ax.invert_yaxis() # invert y axis

    plt.title(f"Aglomerado {cluster['filename']} e Isócrona {iso['filename']} Transladados", fontsize=30)
    plt.xlabel("B-V", fontsize=15)
    plt.ylabel(" V [mag]", fontsize=15)
    plt.legend(fontsize=10)
    plt.show()

    return

def standarize_iso_plots(files_list: list = []) -> plt.figure:
    """
    Test function
    """
    # dict structure aux_dict = {'file_name':{'V':[], '(B-V)':[]}}
    aux_dict = {}
    for file in [random.choice(files_list) for i in range(5)]:
        path_ = os.path.join('topics\R2\clusters', file)

        df = read_iso_file(path=path_)    

        aux_dict[file] = {'V':list(df['V']), '(B-V)': list(df['(B-V)'])}

    return aux_dict

def create_zones(cluster:dict = {}, iso:dict = {},
                 x_tol: float = 0.01, y_tol: float = 0.05) -> dict:
    """
    Function to create the zones on the plot to measure the distance between
    curves
    """
    cluster['V'], cluster['BV'] = translate_data(cluster['V'], cluster['BV'])
    iso['V'], iso['BV'] = translate_data(iso['V'], iso['BV'])

    aux_dict = {'configs':{'x_tol':f"{round(x_tol*100, 3)}%", 'y_tol':f"{round(y_tol*100, 3)}%"}}

    for y0, x0 in zip(iso['V'], iso['BV']):
        Y = round(y0, 5)
        X = round(x0, 5)
        key  = f"{Y:04f},{X:04f}"
        if key in aux_dict:
            #print(f"ERROR AT {key}")
            continue
        else:
            aux_dict[key] = []

        for i in range(len(cluster['V'])):
            x = cluster['BV'][i]
            y = cluster['V'][i]

            accept_x = False
            accept_y = False

            # test_x 
            if x/(x0 or not x0) < (1+x_tol) and x/(x0 or not x0) > (1-x_tol):
                accept_x = True

            # test y
            if y/(y0 or not y0) < (1+y_tol) and y/(y0 or not y0) > (1-y_tol):
                accept_y = True

            if accept_y and accept_x:
                aux_dict[key].append((x, y))

    return aux_dict

def plot_zones(cluster:dict = {}, iso: dict = {},
                x_tol: float = 0.01,
                y_tol: float = 0.05) -> plt.figure:
    """
    Function to show a plot of the created zones for the distance measurements
    """
    def get_data_from_index(indexs:list = []) -> list and list:
        BV = []
        V  = []
        for i in indexs:
            BV.append(cluster['BV'][i])
            V.append(cluster['BV'][i])
        return V, BV

    aux_dict = create_zones(cluster, iso, x_tol, y_tol)
    x_tol_title = aux_dict['configs']['x_tol']
    y_tol_title = aux_dict['configs']['y_tol']
    del aux_dict['configs']

    fig, ax = plt.subplots(figsize=(16,9), dpi=120)

    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

    # plot cluster data
    plt.scatter(cluster['BV'], cluster['V'], marker='o', c='dimgrey', alpha=0.1, label='Cluster Data')
    
    
    n_zones = 0
    for interval in aux_dict.keys():
        
        tup_list = aux_dict[interval]
        if len(tup_list) > 0:
            n_zones +=1
            bv_ = [tup[0] for tup in tup_list]
            v_  = [tup[1] for tup in tup_list]

            plt.scatter(bv_, v_, marker='^', alpha=0.1, c='b')


    
    # plot iso data
    plt.plot(iso['BV'], iso['V'], c='k', ls='-', marker='*', label='Iso Data', alpha=0.8)

    # plot to show infos on legend block
    plt.scatter([], [], label=f'# Zones = {n_zones}', marker='^', c='b')
    # plt.scatter([], [], label=r'$\Delta_X$ = {}'.format(x_tol), marker='.', c='k')
    # plt.scatter([], [], label=r'$\Delta_Y$ = {}'.format(y_tol), marker='.', c='k')


    ax.invert_yaxis()

    plt.title(f"Zones for {cluster['filename']}-{iso['filename']} | $\Delta_X$ = {x_tol}, $\Delta_Y$ = {y_tol}",
                 fontsize=30)

    plt.xlabel("B-V", fontsize=15)
    plt.ylabel(" V [mag]", fontsize=15)
    plt.legend(fontsize=10)
    plt.show()


    return

def get_mean_distance(cluster:dict = {}, iso:dict = {},
                 x_tol: float = 0.01, y_tol: float = 0.05) -> tuple:
    """
    Functio to calculate the mean distance between iso data and cluster, within
    specific zone, data. Returns mean distance and # of zones
    """
    def get_xy_from_key(key: str = ''):
        try:
            y, x = key.split(',')
            return float(y), float(x)
        except:
            print(key)
    
    aux_dict = create_zones(cluster, iso, x_tol, y_tol)
    del aux_dict['configs']


    mean_distance = []
    mean_distance_std = []
    n_zones = 0
    for key in aux_dict.keys():
        distances = []
        tup_list = aux_dict[key]
        size = len(tup_list)  # number of point within the zone
        if size > 0:
            n_zones +=1
            y0, x0 = get_xy_from_key(key)

            bv_ = [tup[0] for tup in tup_list]
            v_  = [tup[1] for tup in tup_list]

            for y_, x_ in zip(v_, bv_):
                delta_y = abs(y_ - y0)
                delta_x = abs(x_ - x0)
                d = np.sqrt(delta_x**2 + delta_y**2)
                distances.append(d)

            mean_distance.append(np.mean(distances))
            mean_distance_std.append(np.std(distances))
        
    return round(np.mean(mean_distance), 7), round(np.mean(mean_distance_std), 7), n_zones

def generate_dataframe_for_iso_files(path: str = '', cluster: dict={}):
    """
    Function to append all the steps. returns a dataframe with information
    of every iso file in _path_
    """
    # expected cluster structure => create_dict_structure()
    files = get_iso_files_list(path)
    df_dict = {'file':[], 'm_dist':[],'std_dist':[], 'n_zones':[]}

    for file in files:
        iso_df = read_iso_file(file)
        iso = create_dict_structure(iso_df, file)

        m_dist, std_dist, n_zones = get_mean_distance(cluster, iso)

        df_dict['file'].append(os.path.basename(file))
        df_dict['m_dist'].append(m_dist)
        df_dict['std_dist'].append(std_dist)
        df_dict['n_zones'].append(n_zones)

        
    df = pd.DataFrame(df_dict)
    df.columns = ['Filename', 'Mean Distance','Mean Distance STD', '# Zones']
    df.sort_values('# Zones', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def translate_cluster_to_iso(cluster: dict={}, iso: dict = {}) -> plt.figure:
    """
    Functio to translate cluster data to iso data. Returns a plot and 
    the translation values
    """
    minV_iso = min(iso['V'])
    minBV_iso = iso['BV'][iso['V'].index(minV_iso)]

    minV_cluster = min(cluster['V'])
    minBV_cluster = cluster['BV'][cluster['V'].index(minV_cluster)]

    mag_cte = minV_cluster - minV_iso
    col_cte = minBV_cluster - minBV_iso

    return mag_cte, col_cte


if __name__ == '__main__':

    for c in [0,3]:
        df_c = read_cluster_file(r'topics\R2\clusters\modR{}_bvr.dat'.format(c))
        clus = create_dict_structure(df_c, 'R{}'.format(c))

        df = generate_dataframe_for_iso_files(path='topics\R2\clusters', cluster=clus)

        df.to_csv('modR{}_bvr_v2.csv'.format(c), index=False)




    #iso2  = create_dict_structure(df_i2, 'bvr_0.008_5000')

    

    

    

    #files = get_iso_files_list('topics\R2\clusters')
    #print(len(files))

    #print(translate_cluster_to_iso(clus, iso))
    #path = 'topics\R2\clusters'

    #df_  = pd.read_csv(r'topics\R2\modR0_bvr_v1.csv')
    #df2  = pd.read_csv(r'topics\R2\modR3_bvr_v1.csv')

    #print(df_.head())
    #print(df2.head())
    #clus = create_dict_structure(df_, 'cluster', )

    #df = generate_dataframe_for_iso_files(path=path, cluster=clus)
    #df.to_csv(r'topics\R2\modR0_bvr_v1.csv', index=False)

    # iso = create_dict_structure(df_i, 'iso')

    # print(get_mean_distance(clus, iso))

    # iso['V'] = iso['V'][:30]
    # iso['BV'] = iso['BV'][:30]

    # clus['BV'] = clus['BV'][:301]
    # clus['V'] = clus['V'][:301]

    # print(get_mean_distance(clus, iso))

    #plot_zones(clus, iso)

    #plot_iso_and_cluster(clus, iso)

    #plot_translated_cluster_data(df['(B-V)'], df['V'], 'modR3_bvr.dat' )

    #d_ = standarize_iso_plots(files)



    