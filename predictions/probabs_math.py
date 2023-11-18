"""
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
import os


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
    return Global_Max_MLE, local_MLEs, lmax, q1, q2, x,y, max_value


def MLE_values(X, sns_output, n):
    """ Get peaks as local MLEs (get also the global max MLE): 
    gets only values. It doesn't pplot anything"""
    l_kde = sns_output.get_lines()[n].get_xydata()   # trying to get the line for the legend
    x,y = l_kde[:,0], l_kde[:,1]
    peaks, _ = find_peaks(y, height=0)
    local_MLEs = []
    for p in peaks:
        MLE = round(x[p],1)
        local_MLEs.append(MLE)
        # l = plt.axvline(x=MLE, color='orange', lw=linewidths[0], ls='dotted')
    max_value = max(y)
    max_index = list(y).index(max_value)
    Global_Max_MLE = round(x[max_index], 2)
    return Global_Max_MLE, local_MLEs, x,y, max_value


def calc_prob_betwen_values(x1,x2):
    ...


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

    ### quantiles
    q11, q12 = np.quantile(x1, percentiles)
    q21, q22 = np.quantile(x2, percentiles)
    q31, q32 = np.quantile(x3, percentiles)
    q41, q42 = np.quantile(x4, percentiles)

    total_counts = sum([len(x1), len(x2), len(x3), len(x4)])

    print('total_counts: ', total_counts)
    print('counts per dataset (1,2,3,4): ', len(x1), len(x2), len(x3), len(x4))
    
    ### probabilidades totales de cada dataset
    p1=round(100*len(x1)/total_counts, 1)
    p2=round(100*len(x2)/total_counts, 1)
    p3=round(100*len(x3)/total_counts, 1)
    p4=round(100*len(x4)/total_counts, 1)
    print('probabilidades porcentuales de x1, x2, x3, x4 =' ,  p1, p2, p3, p4)

    ### count percentiles area
    count1 = round(100*(len(x1[x1>q11])-len(x1[x1>q12]))/total_counts, 2)
    count2 = round(100*(len(x2[x2>q21])-len(x2[x2>q22]))/total_counts, 2)
    count3 = round(100*(len(x3[x3>q31])-len(x3[x3>q32]))/total_counts, 2)
    count4 = round(100*(len(x4[x4>q41])-len(x4[x4>q42]))/total_counts, 2)

    # ### PIE CHART!
    # # Creating dataset
    # efs = ['Ef1', 'Ef2', 'Ef3', 'Ef4']
    # data = [p1, p2, p3, p4]
    # # Creating plot
    # fig = plt.figure(figsize =(10, 7))
    # plt.pie(data, labels = efs, autopct='%.1f%%')
    # # Adding legend
    # plt.legend()
    # # show plot
    # plt.show()
    # ###############

    fig, ax1 = plt.subplots(figsize=figsize)

    # sns_output = sns.histplot((df1, df2, df3, df4), kde=True, stat='percent', alpha=0.00, edgecolor='k', lw=0.0)
    sns_output = sns.histplot((df1, df4), kde=True, stat='percent', alpha=0.00, edgecolor='k', lw=0.0)
    ax1.get_legend().remove()  # remove legend

    p1,p2 = int(percentiles[0]*100), int(percentiles[1]*100)  # change percentiles format for plotting

    # ### add text box
    # import matplotlib.patches as patches
    # rect = patches.Rectangle((0.55,0.75),0.40,0.24, facecolor='grey', alpha=0.2, transform=fig.transFigure)
    # ax1.add_patch(rect)

    # plt.clf()
    ### EF1
    Global_Max_MLE, local_MLEs, lmax, q1, q2, x, y, max_value = MLE(x1, sns_output, 1, color='b', percentiles=percentiles)
    Q1, Q2 = float(q1._xy[0][0]), float(q2._xy[0][0])  # get quantiles from plt object
    mean1 = plt.axvline(x=np.mean(x1), color='b', ls=(0, (30, 100)), lw=1)
    ax1.fill_between(x[(x >= Q1) & (x <= Q2)],y[(x >= Q1) & (x <= Q2)], color="b", alpha=0.2)
    texto = f"Efectividad=1  ---  Máx Probab.: {round(max_value, 1)}% - '{feature}'={Global_Max_MLE}\
        \nPercentiles:  {p1}{'%'} - {p2}{'%'}:     {round(Q1, 2)} - {round(Q2, 2)}"  
    plt.text(0.59, 0.95, texto, color='b', fontsize=13, transform=plt.gca().transAxes, backgroundcolor='w')
    ### add probability
    ypos = (np.max(y)-np.min(y))/2
    plt.text(Global_Max_MLE+Global_Max_MLE*0.02, ypos, f"Prob.: {str(count1)} %", color='b', fontsize=13, backgroundcolor='w')

    ### EF2
    # Global_Max_MLE, local_MLEs, lmax, q1, q2, x, y, max_value = MLE(x2, sns_output,2, color='orange', percentiles=percentiles)
    # Q1, Q2 = float(q1._xy[0][0]), float(q2._xy[0][0])  # get quantiles from plt object
    # ax1.fill_between(x[(x >= Q1) & (x <= Q2)],y[(x >= Q1) & (x <= Q2)], color="orange", alpha=0.2)    
    # texto = f"Efectividad=2  ---  Máx Probab.: {round(max_value, 1)}% - '{feature}'={Global_Max_MLE}\
    #     \nPercentiles:  {p1}{'%'} - {p2}{'%'}:     {round(Q1, 2)} - {round(Q2, 2)}"   
    # plt.text(0.59, 0.89, texto, color='orange', fontsize=13, transform=plt.gca().transAxes, backgroundcolor='w')
    # ### add probability
    # ypos = (np.max(y)-np.min(y))/2
    # plt.text(Global_Max_MLE+Global_Max_MLE*0.02, ypos, f"Prob.: {str(count2)} %", color='orange', fontsize=13, backgroundcolor='w')
    # ### calculate percentiles of Ef=1 for all the others
    # qx2 = stats.percentileofscore(x1, Q1), stats.percentileofscore(x1, Q2)
    # prob_ace_2 = round(qx2[1]-qx2[0], 2)

    ### EF3
    # Global_Max_MLE, local_MLEs, lmax, q1, q2, x, y, max_value = MLE(x3, sns_output,1, color='g', percentiles=percentiles)
    # Q1, Q2 = float(q1._xy[0][0]), float(q2._xy[0][0])  # get quantiles from plt object
    # ax1.fill_between(x[(x >= Q1) & (x <= Q2)],y[(x >= Q1) & (x <= Q2)], color="g", alpha=0.2)
    # texto = f"Efectividad=3  ---  Máx Probab.: {round(max_value, 1)}% - '{feature}'={Global_Max_MLE}\
    #     \nPercentiles:  {p1}{'%'} - {p2}{'%'}:     {round(Q1, 2)} - {round(Q2, 2)}"
    # plt.text(0.59, 0.83, texto, color='g', fontsize=13, transform=plt.gca().transAxes, backgroundcolor='w')
    # ### add probability
    # ypos = (np.max(y)-np.min(y))/2
    # plt.text(Global_Max_MLE+Global_Max_MLE*0.02, ypos, f"Prob.: {str(count3)} %", color='g', fontsize=13, backgroundcolor='w')
    # ### calculate percentiles of Ef=1 for all the others
    # qx3 = stats.percentileofscore(x1, Q1), stats.percentileofscore(x1, Q2)
    # prob_ace_3 = round(qx3[1]-qx3[0], 2)

    ### EF4
    Global_Max_MLE, local_MLEs, lmax, q1, q2, x, y, max_value = MLE(x4, sns_output,0, color='r', percentiles=percentiles)
    Q1, Q2 = float(q1._xy[0][0]), float(q2._xy[0][0])  # get quantiles from plt object
    ax1.fill_between(x[(x >= Q1) & (x <= Q2)],y[(x >= Q1) & (x <= Q2)], color="r", alpha=0.2)
    texto = f"Efectividad=4  ---  Máx Probab.: {round(max_value, 1)}% - '{feature}'={Global_Max_MLE}\
        \nPercentiles:  {p1}{'%'} - {p2}{'%'}:     {round(Q1, 2)} - {round(Q2, 2)}"
    plt.text(0.59, 0.77, texto, color='r', fontsize=13, transform=plt.gca().transAxes, backgroundcolor='w')
    ### add probability
    ypos = (np.max(y)-np.min(y))/2
    plt.text(Global_Max_MLE+Global_Max_MLE*0.02, ypos, f"Prob.: {str(count4)} %", color='r', fontsize=13, backgroundcolor='w')
    ### calculate percentiles of Ef=1 for all the others
    qx4 = stats.percentileofscore(x1, Q1), stats.percentileofscore(x1, Q2)
    prob_ace_4 = round(qx4[1]-qx4[0], 2)
    
    print(f"Para los seleccionados percentiles de Eficiencia 1, estos son los valores de sus poblaciones, para cada subset: ")
    # print(f"prob_Ef._1: {p2-p1}%,   prob_Ef._2: {prob_ace_2}%,   Ef._3: {prob_ace_3}%,   Ef_4: {prob_ace_4}%")
    print(f"prob_Ef._1: {p2-p1}%,   Ef_4: {prob_ace_4}%")
    print(f"Interesa que minimice las densidades de población de cada subset")

    plt.xlabel(feature, fontsize=14)
    q1min, q2max = np.quantile(X_Stacked, (0.0005, 0.9995))
    plt.xlim([q1min, q2max])
    # plt.ylim([0, 170])

    plt.grid(visible=None)
    # tics = [160+(x*5) for x in range(10)]
    # plt.xticks(tics)

    fig.tight_layout()   # Reduce margins in plot

    if plot:
        plt.show()

    if save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:
            logger.info(e)
        file_name = title_+'_'+'.png'
        plt.savefig(join(save_dir, file_name))

    """
    from other article -->  show chart with statistics -->  mean, SD, quantiles 0.025 - 0.975  (95%)
    We also add: median, mode, MaxLikelihook
    """
    ### new probabs
    # NOTE: WE ADD 200 BINS, CAUSE OF THIS: 
    #     "percent: normalize such that bar heights sum to 100"
    #     https://seaborn.pydata.org/generated/seaborn.histplot.html
    plt.clf()
    sns_output = sns.histplot((df1), kde=True, stat='percent', edgecolor='k', lw=0.0, bins=200)
    Global_Max_MLE, local_MLEs, lmax, q1, q2, x1, y1, max_value = MLE(x4, sns_output,0, color='r', percentiles=percentiles)
    area_under_curve(x1,y1)
    print(f"number of points: {len(x1)}, sum y (area?): {sum(y1)}")
    # plt.show()
    
    plt.clf()
    sns_output = sns.histplot((df4), kde=True, stat='percent', edgecolor='k', lw=0.0, bins=200)
    Global_Max_MLE, local_MLEs, lmax, q1, q2, x4, y4, max_value = MLE(x4, sns_output,0, color='r', percentiles=percentiles)
    area_under_curve(x4,y4)
    print(f"number of points: {len(x4)}, sum y (area?): {sum(y4)}")
    # plt.show()
    
    cum1, cum4 = 0, 0 # acumulada
    diff_prob = []
    for xx1, yy1, xx4, yy4 in zip(x1,y1, x4, y4):
        cum1 += yy1
        cum4 += yy4
        dif = cum1-cum4
        print(cum1, cum4, '    dif: ', round(dif,2), '    x: ', round(xx1, 2))

    plt.clf()
    sns_output = sns.histplot((df1), kde=True, stat='percent', edgecolor='k', lw=0.0, bins=200)        
    sns_output = sns.histplot((df4), kde=True, stat='percent', edgecolor='k', lw=0.0, bins=200)
    plt.show()
    
    # TODO: create my own histogram? this one doesn't seem to calculate the area right (not a priority for now, though!)
    #     go forward and calculate probability ratio for each bin
    # TODO: MAIN thing to do now is to calculate the values! con lo que impromimos es suficiente para entender que funciona,
    # falta ahora calcularlo!!!
    
    return X_Stacked # , sns_output


def area_under_curve(x,y):
    from sklearn.metrics import auc
    area = auc(x,y)
    print('\ncomputed AUC using sklearn.metrics.auc: {}'.format(area))
    return area


def get_stats_table(df, features='all', percentiles=(0.1,0.9), save_dir=False):
    """
    from other article -->  show chart with statistics -->  mean, SD, quantiles 0.025 - 0.975  (95%)
    We also add: median, mode, MaxLikelihook    
    """
    from tabulate import tabulate
    vars_dicto = {k:v for v,k in enumerate(df.columns)}
    stats_df = df.describe(percentiles=[.025,0.05, 0.25, 0.5, 0.75, 0.95, 0.975])

    X_Stacked = df.values

    df1 = df[df.Efectividad==1]
    df2 = df[df.Efectividad==2]
    df3 = df[df.Efectividad==3]
    df4 = df[df.Efectividad==4]

    x1 = df1.values
    x2 = df2.values
    x3 = df3.values
    x4 = df4.values

    total_counts = sum([len(x1), len(x2), len(x3), len(x4)])

    print('total_counts: ', total_counts)
    print('counts per dataset (1,2,3,4): ', len(x1), len(x2), len(x3), len(x4))
    
    ### probabilidades totales de cada dataset
    p1=round(100*len(x1)/total_counts, 1)
    p2=round(100*len(x2)/total_counts, 1)
    p3=round(100*len(x3)/total_counts, 1)
    p4=round(100*len(x4)/total_counts, 1)
    print('probabilidades porcentuales de x1, x2, x3, x4 =' ,  p1, p2, p3, p4)


    def get_MLE(vars_dicto, df):
        results={'Efectividad': None}
        counter = 0
        for k,v in vars_dicto.items():
            if v!=0:
                sns_output = sns.histplot((df[k]), kde=True, stat='percent')
                Global_Max_MLE, local_MLEs, x,y, max_value = MLE_values(df.values, sns_output, counter)
                results[k] = Global_Max_MLE
                counter+=1
        plt.clf()
        return results

    results = get_MLE(vars_dicto, df)
    df_results = pd.DataFrame(results,index=['MaxLikelihood'])
    stats_df = stats_df.append(df_results).round(3)
    print()
    logger.info(f"stats_df: ")
    print(tabulate(stats_df, headers=stats_df.columns, tablefmt='fancy_grid'), "\n")

    results = get_MLE(vars_dicto, df1)
    df_results = pd.DataFrame(results,index=['MaxLikelihood'])
    stats_df_1 = df1.describe(percentiles=[.025,0.05, 0.25, 0.5, 0.75, 0.95, 0.975])
    stats_df_1 = stats_df_1.append(df_results).round(3)
    logger.info(f"stats_df_1: ")
    print(tabulate(stats_df_1, headers=stats_df_1.columns, tablefmt='fancy_grid'), "\n")

    results = get_MLE(vars_dicto, df2)
    df_results = pd.DataFrame(results,index=['MaxLikelihood'])
    stats_df_2 = df2.describe(percentiles=[.025,0.05, 0.25, 0.5, 0.75, 0.95, 0.975])
    stats_df_2 = stats_df_2.append(df_results).round(3)
    logger.info(f"stats_df_2: ")
    print(tabulate(stats_df_2, headers=stats_df_2.columns, tablefmt='fancy_grid'), "\n")

    results = get_MLE(vars_dicto, df3)
    df_results = pd.DataFrame(results,index=['MaxLikelihood'])
    stats_df_3 = df3.describe(percentiles=[.025,0.05, 0.25, 0.5, 0.75, 0.95, 0.975])
    stats_df_3 = stats_df_3.append(df_results).round(3)
    logger.info(f"stats_df_2: ")
    print(tabulate(stats_df_3, headers=stats_df_3.columns, tablefmt='fancy_grid'), "\n")

    results = get_MLE(vars_dicto, df4)
    df_results = pd.DataFrame(results,index=['MaxLikelihood'])
    stats_df_4 = df4.describe(percentiles=[.025,0.05, 0.25, 0.5, 0.75, 0.95, 0.975])
    stats_df_4 = stats_df_4.append(df_results).round(3)
    logger.info(f"stats_df_4: ")
    print(tabulate(stats_df_4, headers=stats_df_4.columns, tablefmt='fancy_grid'), "\n")

    # percentile % seems to be the median, but we could add the mode if we feel like it

    ### WE HAVE DONE IT FOR ALL DS. DO THE SAME FOR EACH SUBSET!!!!!!!!!!!!!!!!!!!!! df1, df2, df3, df4
    # sns_output = sns.histplot((df1[k], df2[k], df3[k], df4[k]), kde=True, stat='percent')
    # ...!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

    if save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:
            logger.info(e)
            
        file_pth = join(save_dir, 'stats.xlsx')
        logger.info(f"saving Stats here: '{save_dir}'")
        with pd.ExcelWriter(file_pth) as writer:  
            stats_df.to_excel(writer, sheet_name='general stats')
            stats_df_1.to_excel(writer, sheet_name='stats_Efect_1')
            stats_df_2.to_excel(writer, sheet_name='stats_Efect_2')
            stats_df_3.to_excel(writer, sheet_name='stats_Efect_3')
            stats_df_4.to_excel(writer, sheet_name='stats_Efect_4')

    return stats_df, stats_df_1, stats_df_2, stats_df_3, stats_df_4


if __name__=='__main__':

    #########
    # DS = '/home/javier/mis_proyectos/calculos_Fer/DATAJAVI_V5_deuce.csv'
    # df = pd.read_csv(DS)
    
    #################################################################################################################
    ### MEN
    DS_deuce = '/home/javier/mis_proyectos/calculos_Fer/DATAJAVI_V5_deuce.csv'
    DS_advance = '/home/javier/mis_proyectos/calculos_Fer/DATAJAVI_V5_ad.csv'

    ### WOMEN
    # DS_deuce = '/home/javier/mis_proyectos/calculos_Fer/women_deuce_filtered.csv'
    # DS_advance = '/home/javier/mis_proyectos/calculos_Fer/women_advance_filtered.csv'
    #################################################################################################################
    
    df_deuce   = pd.read_csv(DS_deuce)
    df_advance = pd.read_csv(DS_advance)
    df = df_deuce.append(df_advance)
    df = df.sample(frac=1).reset_index(drop=True)
    # df = df_advance
    # df = df_deuce     
    #########


    feature = 'Serve angle (deg)'
    # Women: 'ANG. IN'  '&(grados)' 'TIME' 'dLinea', 'V(km/h)'
    # Men: 'Efectividad', 'Direction (W,B,T)', 'Speed (Km h-1)', 'Position (m)',
    #    'ZA', 'Net clearance (m)', 'TIME', 'Loss of speed (km h-1)',
    #    'Serve angle (deg)', 'Vertical projection angle (deg)', 'dL (m)'
    save_dir = f"/home/javier/TENNIS_FINAL_RESULTS/{feature}_BOTH"
    
    plot_hist(df, feature=feature, fill_area=False, percentiles=(0.05,0.95), figsize=(15,10), 
        title_=feature, save_dir=False, plot=True)

    # stats_df, stats_df_1, stats_df_2, stats_df_3, stats_df_4 = get_stats_table(df, features='all', 
    #         percentiles=(0.025,0.975), save_dir=save_dir)

    # print('\nWE HAVE DONE IT FOR ALL DS. DO THE SAME FOR EACH SUBSET!!!!!!!!!!!!!!!!!!!!! df1, df2, df3, df4')
    # print("other TODOes: show all statistics in plots as well, show mean and other stats in the histograms. DONE!")   
    # print("\nDO IT FOR DEUCE AND AD.?  SEE LINES 10 & 11 --> train_v5_fastai.py")   
    # print("~/pyenvs/tennis/lib/python3.8/site-packages/fastinference/tabular/shap/interp.py")

    """
    Rafa says:
        Bon día Sr Vives. En principio dijimos todos los primeros saques juntos. 
        Si no es mucho jaleo y se pueden calcular por separado, podemos ver si sale 
        algo interesante en cada uno de los lados...
    """