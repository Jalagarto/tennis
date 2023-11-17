# https://walkwithfastai.com/tab.clas.binary
from fastai.tabular.all import *
from tabulate import tabulate
from os.path import join
import os
import numpy as np
from fastinference.tabular import *  # Shap values
######
import logging
logging.basicConfig(format='%(levelname)s: %(filename)s L:%(lineno)d  -  %(message)s', level=20)
logger = logging.getLogger('tennis')   # logger.debug("yes") # logger.info("no")


class load_data:
    def __init__(self, DS_deuce_pth, DS_advance_pth, ds_type='both'):
        """ ds_type: 'deuce', 'advance', 'both' """
        self.ds_type=ds_type
        if ds_type=='deuce':
            self.df = pd.read_csv(DS_deuce_pth)
        elif ds_type=='advance':
            self.df = pd.read_csv(DS_advance_pth)
        else:
            df_D = pd.read_csv(DS_deuce_pth)
            df_A = df = pd.read_csv(DS_advance_pth)
            self.df = df_D.append(df_A).reset_index()

        print("clases: ", pd.unique(self.df.Efectividad))
        self.subsets()
    
    
    def subsets(self): 
        ### 0. Baseline 1,2,3,4 all 4 classes
        df1,df2,df3,df4 = get_filtered_dfs(self.df)
        self.df_1_2_3_4 = pd.concat([df1, df2, df3, df4])

        ### 1. Entrenar con 1_2_3 vs 4.
        df1,df2,df3,df4 = get_filtered_dfs(self.df)  # reinitialize dfs
        df1.Efectividad = '1,2,3'
        df2.Efectividad = '1,2,3' # 1
        df3.Efectividad = '1,2,3' # 1
        self.df_1_2_3_vs_4 = pd.concat([df1, df2, df3, df4])

        ### 2. Entrenar 2 vs 3_4
        df1, df2, df3, df4 = get_filtered_dfs(self.df)  # refresh dfs
        df3.Efectividad = '3,4'
        df4.Efectividad = '3,4'
        self.df_2_vs_3_4 = pd.concat([df2, df3, df4])

        ### 3. 1vs4,  2vs4, 3vs4  (I.e, 4 vs all)
        df1, df2, df3, df4 = get_filtered_dfs(self.df)  # refresh dfs
        
        logger.info("This is wrong. Oversampling should only be applied to the training data")
        # ---------- oversample df1!!! hardcoded to do this faster --------- #
        df1_oversampled = oversample(df1, std_factor=0.0)
        df1_oversampled = oversample(df1_oversampled, std_factor=0.0)
        self.df_1_4 = pd.concat([df1_oversampled,df4])  ### OVERSAMPLING df1, since df4 is much bigger!
        # ------------------------------------------------------------------ #
        # self.df_1_4 = pd.concat([df1,df4])
        self.df_2_4 = pd.concat([df2,df4])
        self.df_3_4 = pd.concat([df3,df4])

        ### 4. 1vs all --> igual que el pto 3  -->  1vs2, 1vs3, 1vs4
        self.df_1_2 = pd.concat([df1,df2])
        self.df_1_3 = pd.concat([df1,df3])
        # df_1_4 = pd.concat([df3,df4])  # already done

        ### 5. Entrenar con 1_2 vs 3_4.
        df1,df2,df3,df4 = get_filtered_dfs(self.df)
        df1.Efectividad = '1,2'
        df2.Efectividad = '1,2'
        df3.Efectividad = '3,4'
        df4.Efectividad = '3,4'
        self.df_1_2_vs_3_4 = pd.concat([df1, df2, df3, df4])
   
        self.cat_names = ['Direction (W,B,T)']
        # self.cont_names = ['Speed (Km h-1)', 'Position (m)', 'ZA', 'Net clearance (m)', 'TIME', 'Loss of speed (km h-1)', 'Serve angle (deg)', 'Vertical projection angle (deg)', 'dL (m)']   
        # self.cont_names = ['Speed (Km h-1)', 'Position (m)', 'ZA', 'Net clearance (m)', 'Loss of speed (km h-1)', 'Vertical projection angle (deg)', 'dL (m)']  # 'TIME', 'Speed (Km h-1)',  ##### ****** here!!!!*****
        
        # breakpoint()   # important here to select one or the other!
        self.cont_names = ['Speed (Km h-1)', 'Position (m)', 'Net clearance (m)', 'Loss of speed (km h-1)', 'Vertical projection angle (deg)', 'Serve angle (deg)', 'dL (m)'] 
        # self.cont_names = ['Speed (Km h-1)', 'Position (m)', 'Net clearance (m)', 'Loss of speed (km h-1)', 'Vertical projection angle (deg)', 'dL (m)']
        
        ### from correlation matriz: delete --> 'ZA' (keep ANG.IN), delete TIME, keep V(km/h, & dL (m) (keep Serve angle (deg)), 
        # then delete also ... Direction (W,B,T), since it is a very bad variable
        self.y_names = ['Efectividad']           
        
        # ***APScoreBinary:*** Average Precision for single-label binary classification problems  
        # ***APScoreMulti:***  Average Precision for multi-label classification problems        



def get_filtered_dfs(df):
    df1 = df[df.Efectividad==1].copy() # ace
    df2 = df[df.Efectividad==2].copy() # 2 golpes, falla el restador
    df3 = df[df.Efectividad==3].copy() # 3 el sacador gana con 4 golpes o menos
    df4 = df[df.Efectividad==4].copy() # lo gana el sacador, pero más de 4 golpes. No tiene influencia --> todos los que no son 1, 2 o 3
    ### los 3 primeros son efectivos. I.e., el saque tiene influencia importante.
    return df1, df2, df3, df4


def oversample(df, target_col='Efectividad', std_factor=0.01):
    """
    Note: the oversampling method improves the predictions 
        but seems to spoil the feature importance algorithm when using std_factor>0
        I think it is not smart to do it linear, we should add gaussian noise instead.
        So for the moment just duplicate the samples of df1, which has less data    
    """
    # coefficients = np.random.random(len(df))*std_factor
    # coefficients = coefficients.reshape(len(df),1)
    # new_array = coefficients*df.values+df.values
    mean_target_col = df[target_col].mean()
    noise_coefficients = np.random.normal(loc=0.0, scale=std_factor, size=df.shape)
    new_array = noise_coefficients*df.values+df.values
    new_df = pd.DataFrame(new_array, columns = df.columns)
    new_df[target_col] = mean_target_col
    df = df.append(new_df)
    # df = df.sample(frac=1).reset_index(drop=True)
    logger.info(f"by adding a basic data augmentation we have improved by almost 2% the accuracy")
    return df


def train(df, final_epochs, metric, patience, cat_names, cont_names, y_names, seed=0):
    """metric: metric to monitor at last training"""
    
    ### Define PREPROCESSING STEPS
    cat = Categorify()
    to = TabularPandas(df, cat, cat_names)
    cats = to.procs.categorify
    norm = Normalize()
    
    splits = RandomSplitter(valid_pct=0.3, seed=seed)(range_of(df))
    procs = [Categorify, FillMissing, Normalize]
    
    procs = [Categorify, FillMissing, Normalize]
    y_names = 'Efectividad'
    y_block = CategoryBlock()
    to = TabularPandas(df, procs=procs, cat_names=cat_names, cont_names=cont_names,
                       y_names=y_names, y_block=y_block, splits=splits)
    
    ### LOAD DATA Efficiently (using batches in data loaders)
    dls = to.dataloaders(bs=1024)
    
    # learn = tabular_learner(dls, [256, 128, 128, 64], loss_func=FocalLossFlat(), metrics=[accuracy]) # loss_func=CrossEntropyLossFlat(),  256, 128, 128, 64
    # learn = tabular_learner(dls, [200,100], metrics=[accuracy, BalancedAccuracy(), Recall(), Precision(), APScoreMulti()]) # loss_func=CrossEntropyLossFlat(),  256, 128, 128, 64
    learn = tabular_learner(dls, [256, 128, 128, 128, 64], loss_func=CrossEntropyLossFlat(), metrics=[accuracy, BalancedAccuracy()]) # loss_func=CrossEntropyLossFlat(),  256, 128, 128, 64
    
    ###
    # lr = 9e-3
    lr = learn.lr_find(show_plot=False).valley
    learn.lr = lr/5
    print("\n\nCalculated Lrt: ", lr, "using: ", learn.lr, "\n")
    learn.fit_one_cycle(1, slice(lr/(2.6**4),lr))#, moms=(0.8,0.7))
    learn.freeze_to(-2)
    learn.fit_one_cycle(3, slice(lr/(2.6**4),lr))
    learn.unfreeze()
    
    learn.lr = learn.lr*5
    print("\nLearning Rate: ", learn.lr)
    learn.fit_one_cycle(3, slice(lr/(2.6**4),lr))
    
    keep_path = learn.path
    learn.path = Path('/home/javier/mis_proyectos/')
    
    # set the model path to a writeable directory. If you don't do this, the code will produce an error on Gradient
    # learn.path = Path(model_path)
    learn.fit_one_cycle(5, slice(1e-4, 8e-3), 
                        cbs=[EarlyStoppingCallback(monitor='balanced_accuracy_score', min_delta=0.0001, patience=3),
                        SaveModelCallback(monitor='balanced_accuracy_score', min_delta=0.01)])
    
    learn.fit_one_cycle(final_epochs, slice(8e-4, 8e-3),
                        cbs=[EarlyStoppingCallback(monitor=metric, min_delta=0.001, patience=patience),
                        SaveModelCallback(monitor=metric, min_delta=0.01)])
    # learn.fit_one_cycle(12, slice(5e-4, 5e-3))
    
    ### reset the model path
    learn.path = keep_path
    
    interp = ClassificationInterpretation.from_learner(learn)
    # interp.plot_confusion_matrix()
    print("\n### Classification Report: ### ")
    classif_report_dicto = interp.print_classification_report()
    confusion_matrix = interp.confusion_matrix()
    print("### Confusion Matrix: ###\n", confusion_matrix)
    return learn, interp, splits, classif_report_dicto, confusion_matrix


### FEATURE IMPORTANCE:
class PermutationImportance():
    "Calculate and plot the permutation importance"
    def __init__(self, learn:Learner, results_dir, df=None, split_n=0, bs=None, 
            title=False, plot=True, save=False, store_dir=False):  # title='datasets'
        "Initialize with a test dataframe, a learner, and a metric"
        self.results_dir = results_dir
        self.learn = learn
        self.df = df
        bs = bs if bs is not None else learn.dls.bs
        if self.df is not None:
            self.dl = learn.dls.test_dl(self.df, bs=bs)
        else:
            self.dl = learn.dls[1]
        self.x_names = learn.dls.x_names.filter(lambda x: '_na' not in x)
        self.na = learn.dls.x_names.filter(lambda x: '_na' in x)
        self.y = learn.dls.y_names
        self.results = self.calc_feat_importance()
        self.results_df = self.ord_dic_to_df(self.results)
        if plot:
            self.plot_importance(self.results_df, title, split_n, plot=True)
            ### disabled for the moment, since we store it as a dicto
        if save:
            self.plot_importance(self.results_df, title, split_n, plot=False, save=True)

    
    def measure_col(self, name:str):
        "Measures change after column shuffle"
        col = [name]
        if f'{name}_na' in self.na: col.append(name)
        orig = self.dl.items[col].values
        perm = np.random.permutation(len(orig))
        self.dl.items[col] = self.dl.items[col].values[perm]
        metric = self.learn.validate(dl=self.dl)[1]
        self.dl.items[col] = orig
        return metric

    
    def calc_feat_importance(self):
        "Calculates permutation importance by shuffling a column on a percentage scale"
        print('Getting base error')
        base_error = self.learn.validate(dl=self.dl)[1]
        self.importance = {}
        pbar = progress_bar(self.x_names)
        print('Calculating Permutation Importance')
        for col in pbar:
            self.importance[col] = self.measure_col(col)
        for key, value in self.importance.items():
            self.importance[key] = (base_error-value)/base_error #this can be adjusted
        return OrderedDict(sorted(self.importance.items(), key=lambda kv: kv[1], reverse=True))
    

    def ord_dic_to_df(self, dict:OrderedDict):
        return pd.DataFrame([[k, v] for k, v in dict.items()], columns=['feature', 'importance'])
    

    def plot_importance(self, df:pd.DataFrame, title, split_n, limit=20, asc=False, 
                        plot=False, save=False, **kwargs):
        "Plot importance with an optional limit to how many variables shown"
        df_copy = df.copy()
        df_copy['feature'] = df_copy['feature'].str.slice(0,25)
        df_copy = df_copy.sort_values(by='importance', 
                                      ascending=asc)[:limit].sort_values(by='importance', 
                                                                         ascending=not(asc))
        ax = df_copy.plot.barh(x='feature', y='importance', sort_columns=True, **kwargs)
        for p in ax.patches:
            ax.annotate(f'{p.get_width():.4f}', ((p.get_width() * 1.005), p.get_y()  * 1.005))
        plt.xlabel('importance of each feature')
        splits_names = {0:'train', 1:'test'}
        if title:
            plt.title(f"feature permutation importance  -  {title} - {splits_names[split_n]}")
        # ax.get_legend().remove()
        if plot:
            plt.show()
        if save:
            os.makedirs(self.results_dir, exist_ok=True)
            fig_pth = join(self.results_dir, f"{title}_{splits_names[split_n]}.png")
            logger.info(f"saving fig here: '{fig_pth}'")
            plt.savefig(fig_pth)


def main(df, data, hyperp, store_dir, split:list=[0], title=False, shap_subset='all', seed=0):
    """
    split has to be 0 or 1 values, where 0 means to calculate feat importance on training data,
        and 1 on Test Data
    title='subsets df1 - df2'
    """    
    print("split has to be 0 or 1 values, where 0 means to calculate feat importance on training data, ...")
    print("...and 1 on Test Data")
    print(f"{'#'*80}")
    try:
        store_dir = join(store_dir,data.ds_type)
    except:
        print('careful, when title is false the dir to store data is just improvised')
        store_dir = join('~/mis_proyectos', 'no_title_results')
    print(f"creating new dir: {store_dir}")
    os.makedirs(store_dir, exist_ok=True)
    base_file = os.path.split(store_dir)[1]
    final_epochs, metric, patience = hyperp.final_epochs, hyperp.metric, hyperp.patience
    learn, interp, splits, classif_report_dicto, confusion_matrix_np = train(df, final_epochs, metric, 
        patience, data.cat_names, data.cont_names, data.y_names, seed=seed)

    np.savetxt(join(store_dir, "confusion_matrix.csv"), np.around(confusion_matrix_np), delimiter=",", fmt="%d")   

    results = {'classif_report': classif_report_dicto}

    Feature_importance = {}
    for i in split:
        print(f"\n\n Calculating feature importance over split: {i}")
        res = PermutationImportance(learn, store_dir, 
                df.iloc[splits[i]], split_n=i, bs=64, title=title, plot=True, save=True)
        print('\n')
        print(tabulate(res.results_df, headers=res.results_df.columns))
        print(f"\n Results dicto: {res.results}")
        print(f"lenght of the splits: {len(splits[i])} \n")
        Feature_importance[str(i)] = res.results
    results['Feature_importance'] = Feature_importance
    plt.close()

    # fastinference - First let's import the interpretability module:
    # https://walkwithfastai.com/SHAP#fastinference
    print("Trying Shap explanation: ")
    plt.close()  # delete old plt objects
    ### shap subset
    if shap_subset=='all':
        df_shap = df
        print(f"using all points for plotting the shap summary")
    else:
        print(f"using {shap_subset} points for plotting the shap summary")
        df_shap = df.iloc[:shap_subset]
    
    exp = ShapInterpretation(learn, df_shap) # .iloc[:1000])
    print('1')
    exp.summary_plot(show=False)
    plt.legend()
    plt.show()
    fig_title = f"{title}_SHAP.png"
    if title:
        plt.title(fig_title)
        plt.savefig(join(store_dir, fig_title))
    else:
        plt.savefig(join(store_dir, 'shap_summary_plot'))
        
    plt.close()
    with open(join(store_dir, f"{base_file}.json"), 'w') as f:
        json.dump(results, f)

    return learn


def train_1_vs_4(data, csvs_dict:dict, root_dir, ds_type, shap_subset='all', seed=0, plot_correlation=False):
    csvs_dict['deuce'] = '/home/javier/mis_proyectos/calculos_Fer/DATAJAVI_V5_deuce.csv'
    csvs_dict['advance'] = '/home/javier/mis_proyectos/calculos_Fer/DATAJAVI_V5_ad.csv'
    data = load_data(DS_deuce, DS_advance, ds_type=ds_type)

#     ### 1. Entrenar con 1_2_3 vs 4.
#     learn = main(data.df_1_2_3_vs_4, data, hyperp, join(root_dir, 'df_1_2_3_vs_4'), split=[0, 1], 
#                 title='df_1_2_3_vs_4', shap_subset=shap_subset, seed=seed)
# 
#     ### 3. 1vs4,  2vs4, 3vs4  (I.e, 4 vs all)
    
    if plot_correlation:
        # plot correlation matrix (to test multicolinearity)
        """
        https://stats.stackexchange.com/questions/519306/why-important-features-does-not-correlated-with-target-variable/519312#519312
        
        1. Correlation measures the strength of a linear relationship. Age appears to have a weak correlation but, 
            the relationship between age and the outcome may not be linear. See the wikipedia entry for correlation 
            for some examples in which x and y are related but the correlation is 0.

        2. I'm not a big fan of correlation. Feature importance via correlation seems to miss a lot of 
            important variables. I demonstrate this in one of my blog posts. Correlation feature selection 
            (which would be akin to what you're doing here) fails to result in superior performance over other 
            methods across 2 real datasets and 1 simulated dataset. I have little confidence in its ability 
            to successfully pick out good predictors (unless those predictors are linearly related to the 
            outcome and not confounded by any other variables).  
            
            
        https://datascience.stackexchange.com/questions/24452/in-supervised-learning-why-is-it-bad-to-have-correlated-features/24453#24453
        
        Numerical stability aside, prediction given by OLS model should not be affected by multicolinearity, 
        as overall effect of predictor variables is not hurt by presence of multicolinearity. It is interpretation 
        of effect of individual predictor variables that are not reliable when multicolinearity is present.  
        """
        import seaborn as sns
        corr = data.df_1_4.corr()
        sns.heatmap(corr, 
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values,
                    annot=True,
                    cmap="viridis")
        plt.show()
    
    learn = main(data.df_1_4, data, hyperp, False, split=[0, 1], 
                title=False, shap_subset=shap_subset, seed=seed)
#     learn = main(data.df_2_4, data, hyperp, join(root_dir, 'df_2_4'), split=[0, 1], 
#                 title='df_2_4', shap_subset=shap_subset, seed=seed)
#     learn = main(data.df_3_4, data, hyperp, join(root_dir, 'df_3_4'), split=[0, 1], 
#                 title='df_3_4', shap_subset=shap_subset, seed=seed)    



if __name__=='__main__':
    import time
    t1 = time.time()
    class hyperp:
        final_epochs, metric, patience = 50, 'balanced_accuracy_score', 10 # 'balanced_accuracy_score', 3, # 'accuracy'
    # class hyperp0:
    #     final_epochs, metric, patience = 100, 'balanced_accuracy_score', 10 # 'balanced_accuracy_score', 3

    root_dir = '/home/javier/mis_proyectos/tennis_results/try2'

    ### MEN
    DS_deuce = '/home/javier/mis_proyectos/calculos_Fer/DATAJAVI_V5_deuce.csv'
    DS_advance = '/home/javier/mis_proyectos/calculos_Fer/DATAJAVI_V5_ad.csv'

    ### WOMEN
    # DS_deuce = '/home/javier/mis_proyectos/calculos_Fer/women_deuce_filtered.csv'
    # DS_advance = '/home/javier/mis_proyectos/calculos_Fer/women_advance_filtered.csv'
    
    csvs_dict = {'deuce': DS_deuce, 'advance': DS_advance}
    
    shap_subset = 'all' # 10 # 'all'
    seed=1
    for ds_type in ['deuce', 'advance', 'both']:
        print('\n\n -----------  ds_type: ------------',  ds_type, '\n\n')
        data = load_data(DS_deuce, DS_advance, ds_type=ds_type)
        train_1_vs_4(data, csvs_dict, root_dir, ds_type, shap_subset=shap_subset, seed=seed)
        print(f"\n\n\n{'#'*80}\n{'#'*80}\n{'#'*80}\n\n\n")
    
    
    t2 = time.time()
    print(f"The E2E process took {round((t2-t1)/60, 2)} minutes.")
    print("Bye")
    
    import sys
    sys.exit(1)
    
    # ### 0. Baseline 1,2,3,4 all 4 classes
    # df1,df2,df3,df4 = get_filtered_dfs(df)
    # df_1_2_3_4 = pd.concat([df1, df2, df3, df4])    
    # main(df_1_2_3_4, hyperp, join(root_dir, 'df_1_2_3_4'), split=[0, 1], title='df_1_2_3_4')

    ### 1. Entrenar con 1_2_3 vs 4.
    # learn = main(data.df_1_2_3_vs_4, data, hyperp, join(root_dir, 'df_1_2_3_vs_4'), split=[0, 1], title='df_1_2_3_vs_4')

    # ### 2. Entrenar 2 vs 3_4
    # main(df_2_vs_3_4, hyperp, join(root_dir, 'df_2_vs_3_4'), split=[0, 1], title='df_2_vs_3_4')

    ### 3. 1vs4,  2vs4, 3vs4  (I.e, 4 vs all)
    learn = main(data.df_1_4, data, hyperp, False, split=[0, 1], title=False)# title='df_1_4')
    # learn = main(data.df_2_4, data, hyperp, join(root_dir, 'df_2_4'), split=[0, 1], title='df_2_4')
    # learn = main(data.df_3_4, data, hyperp, join(root_dir, 'df_3_4'), split=[0, 1], title='df_3_4')

    # ### 4. 1vs all --> igual que el pto 3  -->  1vs2, 1vs3, 1vs4   ...   # df_1_4  already done above
    # main(df_1_2, hyperp, join(root_dir, 'df_1_2'), split=[0, 1], title='df_1_2')
    # main(df_1_3, hyperp, join(root_dir, 'df_1_3'), split=[0, 1], title='df_1_3')

    ### 5. Entrenar con 1_2 vs 3_4.
    # main(df_1_2_vs_3_4, hyperp, join(root_dir, 'df_1_2_vs_3_4'), split=[0, 1], title='df_1_2_vs_3_4')


#     final_epochs, metric, patience = 50, 'balanced_accuracy_score', 3
#     learn, interp, splits = train(df_2_vs_3_4, final_epochs, metric, patience, cat_names, cont_names, y_names)
# 
#     y_val = df_2_vs_3_4.iloc[splits[0]].Efectividad.values
#     X_val = df_2_vs_3_4.iloc[splits[0]][['Direction (W,B,T)','Speed (Km h-1)','Position (m)','ZA','Net clearance (m)','TIME','Loss of speed (km h-1)','Serve angle (deg)','Vertical projection angle (deg)','dL (m)']].values
# 
#     res = PermutationImportance(learn, df_2_vs_3_4.iloc[splits[0]], bs=64)
#     print('\n')
#     print(tabulate(res.results_df, headers=res.results_df.columns))

    print("DO IT FOR DEUCE AND AD.?  SEE LINES 10 & 11")


    msg = """
    Rafa says:
        Bon día Sr Vives. En principio dijimos todos los primeros saques juntos. 
        Si no es mucho jaleo y se pueden calcular por separado, podemos ver si sale 
        algo interesante en cada uno de los lados...
    """

    logger.info(msg)

"""
TODO: Make Report Again
"""
### analisis discriminante --> clustering?

# python train_v5_fastai.py 2>&1 | tee /home/javier/mis_proyectos/tennis_results/try2/train.log


"""
self.cat_names = ['Direction (W,B,T)']
self.cont_names = ['Speed (Km h-1)', 'Position (m)', 'Net clearance (m)', 'Loss of speed (km h-1)', 'Serve angle (deg)', 'Vertical projection angle (deg)']
# self.cont_names = ['Speed (Km h-1)', 'Position (m)', 'Net clearance (m)', 'Loss of speed (km h-1)', 'dL (m)', 'Vertical projection angle (deg)']
"""
