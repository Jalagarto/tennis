"""
Adaptation to arsos_robots_ai/AI_recommendations/stats_resets.ipynb to runable script.

The goal of this module is to understand the model:
    - data distribution, 
    - KDE, 
    - mean, mode, variance, std_dev, etc.
    - Global and locals ML (maximum Likelihood points)
    - etc.

I.e., Understand data distribution through statistics.

Output: Information to output a recommendation value

We compute statistics of True / False class. For the moment for succesful and profitable (or not) resets
"""

### load our Data:
from os.path import join
import numpy as np
import logging
import seaborn as sns
import pandas as pd

import logging
logging.basicConfig(format='%(levelname)s: %(filename)s L:%(lineno)d  -  %(message)s', level=20)
logger = logging.getLogger('tennis')   # logger.debug("yes") # logger.info("no")

import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

import matplotlib.pyplot as plt


def MLE(X, sns_output, n, color='b', percentiles=(0.15,0.85)):
    """ Get peaks as local MLEs (get also the global max MLE): """
    l_kde = sns_output.get_lines()[n].get_xydata()   # trying to get the line for the legend
    x,y = l_kde[:,0], l_kde[:,1]
    q1, q2 = np.quantile(X, percentiles)
    peaks, _ = find_peaks(y, height=0)
    local_MLEs = []
    for p in peaks:
        MLE = round(x[p],1)
        local_MLEs.append(MLE)
        # l = plt.axvline(x=MLE, color='orange', lw=linewidths[0], ls='dotted')
    max_value = max(y)
    max_index = list(y).index(max_value)
    Global_Max_MLE = round(x[max_index], 2)
    lmax = plt.axvline(x=Global_Max_MLE, color=color, ls=(0, (5, 5)))
    q1 = plt.axvline(x=q1, color=color, ls=(0, (5, 30)), lw=1)
    q2 = plt.axvline(x=q2, color=color, ls=(0, (5, 30)), lw=1)
    return Global_Max_MLE, local_MLEs, lmax, q1, q2, x,y


def plot_hist(df, feature='&(grados)', fill_area=True, percentiles=(0.1,0.9), figsize=(15,8), title_='Successful', 
    linewidths=(1,2), save_dir=False, plot=False):
    """ 
    plot hist with quantiles, mode and mean
    wtg is only used in the title in this method
    save_dir: dir to save images --> [False, 'dir_path']
    plot: bool  [True, False]
    """
    vars_dicto = {k:v for v,k in enumerate(df.columns)}

    X_Stacked = df.values[:, vars_dicto[feature]]
    
    df1 = df[df.Efectividad==1][feature]
    df2 = df[df.Efectividad==2][feature]
    df3 = df[df.Efectividad==3][feature]
    df4 = df[df.Efectividad==4][feature]

    df2.Efectividad = 1
    df3.Efectividad = 4
    df_1_2 = pd.concat([df1, df2])
    df_3_4 = pd.concat([df3, df4])

    x1 = df1.values
    x2 = df2.values
    x3 = df3.values
    x4 = df4.values
    x_1_2 = df_1_2.values
    x_3_4 = df_3_4.values

    df1 = df1.rename('Efectividad = 1', inplace=True)
    df2 = df2.rename('Efectividad = 2', inplace=True)
    df3 = df3.rename('Efectividad = 3', inplace=True)
    df4 = df4.rename('Efectividad = 4', inplace=True)
    df_1_2 = df_1_2.rename('Efectividad = 1 & 2', inplace=True)
    df_3_4 = df_3_4.rename('Efectividad = 3 & 4', inplace=True)    

    fig, ax1 = plt.subplots(figsize=figsize)

    # ax1.fill_between(x,y, color="red", alpha=0.2)

    # sns_output = sns.histplot((df1, df2, df3, df4), kde=True, alpha=0.20, edgecolor='k', lw=0.0)
    sns_output = sns.histplot((df1, df2, df3, df4, df_1_2, df_3_4), kde=True, alpha=0.0, edgecolor='k', lw=0.0)
    ax1.get_legend().remove()  # remove legend

    ### add text box
    import matplotlib.patches as patches
    rect = patches.Rectangle((0.74,0.83),0.21,0.14, facecolor='grey', alpha=0.2, transform=fig.transFigure)
    ax1.add_patch(rect)

    # plt.clf()
    Global_Max_MLE, local_MLEs, lmax, q1, q2, x, y = MLE(x_1_2, sns_output,0, color='maroon')
    Q1, Q2 = float(q1._xy[0][0]), float(q2._xy[0][0])  # get quantiles from plt object
    ax1.fill_between(x[(x >= Q1) & (x <= Q2)],y[(x >= Q1) & (x <= Q2)], color="maroon", alpha=0.2)
    plt.text(1.6, 260, f"Efectividad=1_2  ---  Máx Prob: {Global_Max_MLE}", color='maroon', fontsize=12)
    
    Global_Max_MLE, local_MLEs, lmax, q1, q2, x, y = MLE(x_3_4, sns_output,1, color='m')
    Q1, Q2 = float(q1._xy[0][0]), float(q2._xy[0][0])  # get quantiles from plt object
    ax1.fill_between(x[(x >= Q1) & (x <= Q2)],y[(x >= Q1) & (x <= Q2)], color="m", alpha=0.2)
    plt.text(1.6, 250, f"Efectividad=3_4  ---  Máx Prob: {Global_Max_MLE}", color='m', fontsize=12)

#     # plt.clf()
#     Global_Max_MLE, local_MLEs, lmax, q1, q2, x, y = MLE(x4, sns_output,0, color='r')
#     Q1, Q2 = float(q1._xy[0][0]), float(q2._xy[0][0])  # get quantiles from plt object
#     ax1.fill_between(x[(x >= Q1) & (x <= Q2)],y[(x >= Q1) & (x <= Q2)], color="r", alpha=0.0)
    plt.text(1.6, 240, f"Efectividad=4  ---  Máx Prob: {Global_Max_MLE}", color='r', fontsize=11)
#     
#     Global_Max_MLE, local_MLEs, lmax, q1, q2, x, y = MLE(x3, sns_output,1, color='g')
#     Q1, Q2 = float(q1._xy[0][0]), float(q2._xy[0][0])  # get quantiles from plt object
#     ax1.fill_between(x[(x >= Q1) & (x <= Q2)],y[(x >= Q1) & (x <= Q2)], color="g", alpha=0.0)
    plt.text(1.6, 230, f"Efectividad=3  ---  Máx Prob: {Global_Max_MLE}", color='g', fontsize=11)
#     
#     Global_Max_MLE, local_MLEs, lmax, q1, q2, x, y = MLE(x2, sns_output,2, color='orange')
#     Q1, Q2 = float(q1._xy[0][0]), float(q2._xy[0][0])  # get quantiles from plt object
#     ax1.fill_between(x[(x >= Q1) & (x <= Q2)],y[(x >= Q1) & (x <= Q2)], color="orange", alpha=0.0)    
    plt.text(1.6, 220, f"Efectividad=2  ---  Máx Prob: {Global_Max_MLE}", color='orange', fontsize=11)
# 
#     Global_Max_MLE, local_MLEs, lmax, q1, q2, x, y = MLE(x1, sns_output,3, color='b')
#     Q1, Q2 = float(q1._xy[0][0]), float(q2._xy[0][0])  # get quantiles from plt object
#     ax1.fill_between(x[(x >= Q1) & (x <= Q2)],y[(x >= Q1) & (x <= Q2)], color="b", alpha=0.0)    
    plt.text(1.6, 210, f"Efectividad=1  ---  Máx Prob: {Global_Max_MLE}", color='b', fontsize=11)
    
    plt.xlabel(feature)
    # plt.xlim([120, 240])
    # plt.ylim([0, 200])


    plt.grid(visible=None)
    # tics = [160+(x*5) for x in range(10)]
    # plt.xticks(tics)

    fig.tight_layout()   # Reduce margins in plot

    if plot:
        plt.show()

    if save_dir:
        file_name = title_+'_'+'.png'
        plt.savefig(join(save_dir, file_name))

    return X_Stacked # , sns_output



if __name__=='__main__':

    #########
    DS = '/home/javier/mis_proyectos/calculos_Fer/DATAJAVI_V5_deuce.csv'
    df = pd.read_csv(DS)
    #########

    plot_hist(df, feature='dLinea', fill_area=False, percentiles=(0.05,0.95), figsize=(15,10), 
        title_='kde', plot=True)

    logger.info("TODO: do the same with gausian KDE. Don't plot the bars, but plot percentiles areas in between!")