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
    return Global_Max_MLE, local_MLEs, lmax, q1, q2


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

    x1 = df1.values
    x2 = df2.values
    x3 = df3.values
    x4 = df4.values

    df1 = df1.rename('Efectividad = 1', inplace=True)
    df2 = df2.rename('Efectividad = 2', inplace=True)
    df3 = df3.rename('Efectividad = 3', inplace=True)
    df4 = df4.rename('Efectividad = 4', inplace=True)    

    fig, ax1 = plt.subplots(figsize=figsize)

    # ax1.fill_between(x,y, color="red", alpha=0.2)
    # plt.clf()
    sns_output = sns.histplot((df1, df2, df3, df4), kde=True, alpha=0.1, legend=True, lw=0.0)
    Global_Max_MLE, local_MLEs, lmax, q1, q2 = MLE(x4, sns_output,0, color='r')
    plt.text(8,175, f"Máx Prob: {Global_Max_MLE}", color='r')

    Global_Max_MLE, local_MLEs, lmax, q1, q2 = MLE(x3, sns_output,1, color='g')
    plt.text(8,180, f"Máx Prob: {Global_Max_MLE}", color='g')

    Global_Max_MLE, local_MLEs, lmax, q1, q2 = MLE(x2, sns_output,2, color='orange')
    plt.text(8,185, f"Máx Prob: {Global_Max_MLE}", color='orange')

    Global_Max_MLE, local_MLEs, lmax, q1, q2 = MLE(x1, sns_output,3, color='b')
    plt.text(8,190, f"Máx Prob: {Global_Max_MLE}", color='b')

    # plt.grid(visible=None)
    tics = [x*0.5 for x in range(18)]
    plt.xticks(tics)

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

    plot_hist(df, fill_area=False, percentiles=(0.05,0.95), figsize=(15,10), 
        title_='kde', plot=True)


    logger.info("TODO: do the same with gausian KDE. Don't plot the bars, but plot percentiles areas in between!")