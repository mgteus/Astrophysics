from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
import scipy.stats as ss
from sympy import Si

from shapley_aux_funcs import read_dat_file, find_missing_values, fill_missing_values, set_values_on_df

def create_XYZ_plots(X:list = [], Y: list = [], Z: list = []) -> plt.figure:
    """ 
    Function to create the plots for X, Y and Z distribuitions
    """
    x_mean = np.mean(X)
    x_std  = np.std(X)

    y_mean = np.mean(Y)
    y_std  = np.std(Y)

    z_mean = np.mean(Z)
    z_std  = np.std(Z)

    # X-Y plot
    fig, ax = plt.subplots(figsize=(16,9), dpi=120)

    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

    plt.title('Distribuição de X x Y', fontsize=30)
    plt.scatter(X, Y, marker='^', label='Data')
    plt.hlines(y_mean, xmin=min(X)-100, xmax=max(X)+100, ls='--', color='k',
                 alpha=0.2, 
                 label=r'$\langle{Y}\rangle$'+f" = {round(x_mean,3)}$\pm${round(x_std, 3)}")
    plt.vlines(x_mean, ymax=max(Y)+100, ymin=min(Y)-100, ls='--', color='k',
                 alpha=0.2, 
                 label=r'$\langle{X}\rangle$'+f" = {round(y_mean,3)}$\pm${round(y_std, 3)}")
    plt.xlabel("X (Kpc)", fontsize=15)
    plt.ylabel("Y (Kpc)", fontsize=15, rotation=0)
    plt.xlim(min(X)-10, max(X)+10)
    plt.ylim(min(Y)-10, max(Y)+10)
    plt.legend(fontsize=20)
    plt.show()
    # X-Z plot
    fig, ax = plt.subplots(figsize=(16,9), dpi=120)

    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

    plt.title('Distribuição de X x Z', fontsize=30)
    plt.scatter(X, Z, marker='^', label='Data')
    plt.hlines(z_mean, xmin=min(X)-100, xmax=max(X)+100, ls='--', color='k', 
                alpha=0.2, 
                label=r'$\langle{Z}\rangle$'+f" = {round(z_mean,3)}$\pm${round(z_std, 3)}")
    plt.vlines(x_mean, ymax=max(Z)+100, ymin=min(Z)-100, ls='--', color='k', 
                alpha=0.2, 
                label=r'$\langle{X}\rangle$'+f" = {round(x_mean,3)}$\pm${round(x_std, 3)}")
    plt.xlabel("X (Kpc)", fontsize=15)
    plt.ylabel("Z (Kpc)", fontsize=15, rotation=0)
    plt.xlim(min(X)-10, max(X)+10)
    plt.ylim(min(Z)-10, max(Z)+10)
    plt.legend(fontsize=20)
    plt.show()

    # Y-Z plot
    fig, ax = plt.subplots(figsize=(16,9), dpi=120)

    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

    plt.title('Distribuição de Y x Z', fontsize=30)
    plt.scatter(Y, Z, marker='^', label='Data')
    plt.hlines(z_mean, xmin=min(Y)-100, xmax=max(Y)+100, ls='--', color='k', 
                alpha=0.2, 
                label=r'$\langle{Z}\rangle$'+f" = {round(z_mean,3)}$\pm${round(z_std, 3)}")
    plt.vlines(y_mean, ymax=max(Z)+100, ymin=min(Z)-100, ls='--', color='k', 
                alpha=0.2, 
                label=r'$\langle{Y}\rangle$'+f" = {round(y_mean,3)}$\pm${round(y_std, 3)}")
    plt.xlabel("Y (Kpc)", fontsize=15)
    plt.ylabel("Z (Kpc)", fontsize=15, rotation=0)
    plt.xlim(min(Y)-10, max(Y)+10)
    plt.ylim(min(Z)-10, max(Z)+10)
    plt.legend(fontsize=20)
    plt.show()

    std_menos = np.sqrt((x_mean-x_std)**2 + (y_mean-y_std)**2 + (z_mean-z_mean)**2)
    std_maiss = np.sqrt((x_mean+x_std)**2 + (y_mean+y_std)**2 + (z_mean+z_mean)**2)
    print(f"Distancia Sol-CG = {np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)} +- {(std_maiss+std_menos)/2}")
    return


def create_gaussian_fit(data:list = [], var: str ='') -> plt.figure:
    """
    Function to fit a gaussian to a list of data points
    """

    data_mean = np.mean(data)
    data_std  = np.std(data)
    data_var  = np.var(data)

    # data's histogram

    fig, ax = plt.subplots(figsize=(16,9), dpi=120)

    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)


    new_data = [d for d in data if abs(d) <= (abs(data_mean)+abs(1.5*data_std))]
    new_data_mean = np.mean(new_data)
    new_data_std  = np.std(new_data)
    new_data_var  = np.var(new_data)

    plt.title(f"Fit Gaussiano para {var}", fontsize=30)
    x_ = np.linspace(np.min(data),np.max(data),len(data))
    #y_ = 1.0/np.sqrt(2*np.pi*data_var)*np.exp(-0.5*(x_-data_mean)**2/data_var)
    #print(y_)
    #y_one_sigma = [y for y in y_ if abs(y) <= (data_mean+10*data_std)]
    _, bins, _ = plt.hist(new_data, label = 'Data [$\mu \pm \sigma$]', bins=25, density=0)
    mu, sigma = scipy.stats.norm.fit(new_data)

    best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)

    plt.plot(bins, 250*best_fit_line,'r--', label='Gaussian Fit '+f"($\mu_g$ = {round(new_data_mean, 3)}, $\sigma_g$ = {round(new_data_std, 5)})", lw=3)
    #plt.plot(x_,y_,'r--', label='Gaussian Fit '+f"($\mu$ = {round(data_mean, 3)}, $\sigma$ = {round(data_std, 5)})", lw=3)
    plt.ylabel('Freq',  fontsize=20)
    #plt.xlim(-20, 40)
    plt.xlabel(f"{var} (Kpc)",  fontsize=20)
    
    
    plt.legend(fontsize=20)
    #plt.show()


    print(f"media_antes = {data_mean}, sigma_antes = {data_std}")
    print(f"media_ajustada = {round(new_data_mean, 4)}, sigma_ajustado = {round(new_data_std, 4)}")
    return new_data

def create_3D_plot(X:list = [], Y: list = [], Z: list = []) -> plt.figure:
    """
    Function to create the 3D plot for the giver arguments
    """
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    z_mean = np.mean(Z)

    x_ = [x_mean for i in range(int(len(X)))]
    y_ = [y_mean for i in range(int(len(Y)))]
    z_ = [z_mean for i in range(int(len(Z)))]
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')


    ax.set_title("Gráfico de X, Y e Z")
    ax.scatter3D(X, Y, Z, label='Data', alpha=0.1, c='k')
    ax.plot3D(np.linspace(min(X), max(X), len(X)), y_, z_, c='r', ls='--', lw=1, alpha=0.5,)
    ax.plot3D(x_, np.linspace(min(Y), max(Y), len(Y)), z_, c='r', ls='--', lw=1, alpha=0.5,)
    ax.plot3D(x_, y_, np.linspace(min(Z), max(Z), len(Z)), c='r', ls='--', lw=1, alpha=0.5, label=r'$\langle{X}\rangle,\langle{Y}\rangle,\langle{Z}\rangle$')
    ax.scatter3D(x_mean, y_mean, z_mean, 'r', c='r',s=50, label='CG')

    ax.set_xlabel('X (Kpc)')
    ax.set_ylabel('Y (Kpc)')
    ax.set_zlabel('Z (Kpc)')

    # Data for three-dimensional scattered points

    

    plt.legend()
    plt.show()  

    return

def create_aitoff_plot(X:list =[], Y:list=[]) -> plt.figure:
    """
    Function to create the aitoff's projection for the giver arguments
    """

    def fix_long(x):
        if x < 180:
            return x 
        else:
            return x - 360


    
    X = [fix_long(x)*np.pi/180 for x in X]
    Y = [y*np.pi/180 for y in Y]
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    plt.figure()
    plt.subplot(projection="aitoff")
    plt.scatter(X, Y, lw=0, marker='o', label='Data')
    plt.scatter(x_mean, y_mean, c='r', lw=0, marker='o', label='GC')

    plt.title("Projeção de Aitoff")
    plt.grid(True)

    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    return 

if __name__ == "__main__":
    df = pd.read_csv("data_completo.csv")

    
    #print(df.head(30))
    #create_XYZ_plots(df['X'], df['Y'], df['Z'])

    #x = create_gaussian_fit(df['X'], 'X')

    #y = create_gaussian_fit(df['Y'], 'Y')

    #z = create_gaussian_fit(df['Z'], 'Z')

    #create_3D_plot(x, y, z)

    #create_aitoff_plot(df['L'], df['B'])