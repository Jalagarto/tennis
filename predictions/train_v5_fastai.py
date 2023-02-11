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


DS = '/home/javier/mis_proyectos/calculos_Fer/DATAJAVI_V5_deuce.csv'
# DS = '/home/javier/mis_proyectos/calculos_Fer/DATAJAVI_V5_ad.csv'

### FASTAI:
df = pd.read_csv(DS)

print("clases: ", pd.unique(df.Efectividad))

def get_filtered_dfs(df):
    df1 = df[df.Efectividad==1].copy() # ace
    df2 = df[df.Efectividad==2].copy() # 2 golpes, falla el restador
    df3 = df[df.Efectividad==3].copy() # 3 el sacador gana con 4 golpes o menos
    df4 = df[df.Efectividad==4].copy() # lo gana el sacador, pero más de 4 golpes. No tiene influencia --> todos los que no son 1, 2 o 3
    ### los 3 primeros son efectivos. I.e., el saque tiene influencia importante.
    return df1, df2, df3, df4


##################################################################################################################################
###### Note that this should just be a class --> no time to code properly. Spend some time in the future to make it better #######
##################################################################################################################################

### 0. Baseline 1,2,3,4 all 4 classes
df1,df2,df3,df4 = get_filtered_dfs(df)
df_1_2_3_4 = pd.concat([df1, df2, df3, df4])

### 1. Entrenar con 1_2_3 vs 4.
df1,df2,df3,df4 = get_filtered_dfs(df)  # reinitialize dfs
df1.Efectividad = '123'
df2.Efectividad = '123' # 1
df3.Efectividad = '123' # 1
df_1_2_3_vs_4 = pd.concat([df1, df2, df3, df4])

### 2. Entrenar 2 vs 3_4
df1, df2, df3, df4 = get_filtered_dfs(df)  # refresh dfs
df3.Efectividad = 4
df_2_vs_3_4 = pd.concat([df2, df3, df4])

### 3. 1vs4,  2vs4, 3vs4  (I.e, 4 vs all)
df1, df2, df3, df4 = get_filtered_dfs(df)  # refresh dfs
df_1_4 = pd.concat([df1,df4])
df_2_4 = pd.concat([df2,df4])
df_3_4 = pd.concat([df3,df4])

### 4. 1vs all --> igual que el pto 3  -->  1vs2, 1vs3, 1vs4
df_1_2 = pd.concat([df1,df2])
df_1_3 = pd.concat([df1,df3])
# df_1_4 = pd.concat([df3,df4])  # already done

### 5. Entrenar con 1_2 vs 3_4.
df1,df2,df3,df4 = get_filtered_dfs(df)
df2.Efectividad = 1
df3.Efectividad = 4
df_1_2_vs_3_4 = pd.concat([df1, df2, df3, df4])


# ***APScoreBinary:*** Average Precision for single-label binary classification problems  
# ***APScoreMulti:***  Average Precision for multi-label classification problems

cat_names = ['DIRECCIÓN:1 abierto;2 al cuerpo;3 a la T']
cont_names = ['V(km/h)', '[YA]', 'ZA', 'Znet', 'TIME', 'difV', '&(grados)', 'ANG. IN', 'dLinea']
y_names = ['Efectividad']


def train(df, final_epochs, metric, patience, cat_names, cont_names, y_names):
    """metric: metric to monitor at last training"""
    cat = Categorify()
    to = TabularPandas(df, cat, cat_names)
    cats = to.procs.categorify
    norm = Normalize()
    
    splits = RandomSplitter(valid_pct=0.3)(range_of(df))
    procs = [Categorify, FillMissing, Normalize]
    
    procs = [Categorify, FillMissing, Normalize]
    y_names = 'Efectividad'
    y_block = CategoryBlock()
    to = TabularPandas(df, procs=procs, cat_names=cat_names, cont_names=cont_names,
                       y_names=y_names, y_block=y_block, splits=splits)
    
    dls = to.dataloaders(bs=1024)
    # learn = tabular_learner(dls, [256, 128, 128, 64], loss_func=FocalLossFlat(), metrics=[accuracy]) # loss_func=CrossEntropyLossFlat(),  256, 128, 128, 64
    # learn = tabular_learner(dls, [200,100], metrics=[accuracy, BalancedAccuracy(), Recall(), Precision(), APScoreMulti()]) # loss_func=CrossEntropyLossFlat(),  256, 128, 128, 64
    learn = tabular_learner(dls, [200,100], metrics=[accuracy, BalancedAccuracy()]) # loss_func=CrossEntropyLossFlat(),  256, 128, 128, 64
    
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
            title='datasets', plot=True, save=False, store_dir=False):
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
        plt.title(f"feature permutation importance  -  {title} - {splits_names[split_n]}")
        # ax.get_legend().remove()
        if plot:
            plt.show()
        if save:
            os.makedirs(self.results_dir, exist_ok=True)
            fig_pth = join(self.results_dir, f"{title}_{splits_names[split_n]}.png")
            logger.info(f"saving fig here: '{fig_pth}'")
            plt.savefig(fig_pth)


def main(df, hyperp, store_dir, split:list=[0], title='subsets df1 - df2'):
    """
    split has to be 0 or 1 values, where 0 means to calculate feat importance on training data,
        and 1 on Test Data
    """
    print("split has to be 0 or 1 values, where 0 means to calculate feat importance on training data, ...")
    print("...and 1 on Test Data")
    print(f"{'#'*50}")
    print(f"creating new dir: {store_dir}")
    os.makedirs(store_dir, exist_ok=True)
    base_file = os.path.split(store_dir)[1]
    final_epochs, metric, patience = hyperp.final_epochs, hyperp.metric, hyperp.patience
    learn, interp, splits, classif_report_dicto, confusion_matrix_np = train(df, final_epochs, metric, 
        patience, cat_names, cont_names, y_names)

    np.savetxt(join(store_dir, "confusion_matrix.csv"), np.around(confusion_matrix_np), delimiter=",", fmt="%d")   

    results = {'classif_report': classif_report_dicto}

    Feature_importance = {}
    for i in split:
        print(f"\n\n Calculating feature importance over split: {i}")
        res = PermutationImportance(learn, store_dir, 
                df.iloc[splits[i]], split_n=i, bs=64, title=title, plot=False, save=True)
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
    exp = ShapInterpretation(learn, df) # .iloc[:100])
    print('1')
    exp.summary_plot(show=False)
    fig_title = f"{title}_SHAP.png"
    plt.title(fig_title)
    plt.savefig(join(store_dir, fig_title))
    plt.close()
    with open(join(store_dir, f"{base_file}.json"), 'w') as f:
        json.dump(results, f)

    return learn


if __name__=='__main__':

    class hyperp:
        final_epochs, metric, patience = 50, 'accuracy', 3 # 'balanced_accuracy_score', 3
    # class hyperp0:
    #     final_epochs, metric, patience = 100, 'balanced_accuracy_score', 10 # 'balanced_accuracy_score', 3

    root_dir = '/home/javier/mis_proyectos/tennis_results/try2'

    # ### 0. Baseline 1,2,3,4 all 4 classes
    # df1,df2,df3,df4 = get_filtered_dfs(df)
    # df_1_2_3_4 = pd.concat([df1, df2, df3, df4])    
    # main(df_1_2_3_4, hyperp, join(root_dir, 'df_1_2_3_4'), split=[0, 1], title='df_1_2_3_4')

    ### 1. Entrenar con 1_2_3 vs 4.
    learn = main(df_1_2_3_vs_4, hyperp, join(root_dir, 'df_1_2_3_vs_4'), split=[0, 1], title='df_1_2_3_vs_4')

    # ### 2. Entrenar 2 vs 3_4
    # main(df_2_vs_3_4, hyperp, join(root_dir, 'df_2_vs_3_4'), split=[0, 1], title='df_2_vs_3_4')

    ### 3. 1vs4,  2vs4, 3vs4  (I.e, 4 vs all)
    learn = main(df_1_4, hyperp, join(root_dir, 'df_1_4'), split=[0, 1], title='df_1_4')
    learn = main(df_2_4, hyperp, join(root_dir, 'df_2_4'), split=[0, 1], title='df_2_4')
    learn = main(df_3_4, hyperp, join(root_dir, 'df_3_4'), split=[0, 1], title='df_3_4')

    # ### 4. 1vs all --> igual que el pto 3  -->  1vs2, 1vs3, 1vs4   ...   # df_1_4  already done above
    # main(df_1_2, hyperp, join(root_dir, 'df_1_2'), split=[0, 1], title='df_1_2')
    # main(df_1_3, hyperp, join(root_dir, 'df_1_3'), split=[0, 1], title='df_1_3')

    ### 5. Entrenar con 1_2 vs 3_4.
    # main(df_1_2_vs_3_4, hyperp, join(root_dir, 'df_1_2_vs_3_4'), split=[0, 1], title='df_1_2_vs_3_4')


#     final_epochs, metric, patience = 50, 'balanced_accuracy_score', 3
#     learn, interp, splits = train(df_2_vs_3_4, final_epochs, metric, patience, cat_names, cont_names, y_names)
# 
#     y_val = df_2_vs_3_4.iloc[splits[0]].Efectividad.values
#     X_val = df_2_vs_3_4.iloc[splits[0]][['DIRECCIÓN:1 abierto;2 al cuerpo;3 a la T','V(km/h)','[YA]','ZA','Znet','TIME','difV','&(grados)','ANG. IN','dLinea']].values            
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
