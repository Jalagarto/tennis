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
        ### shuffle df:
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        print("clases: ", pd.unique(self.df.Efectividad))
        self.subsets()
    
    
    def oversample(self, df, target_col='Efectividad', std_factor=0.1):
        coefficients = np.random.random(len(df))*0.01
        coefficients = coefficients.reshape(len(df),1)
        mean_target_col = df[target_col].mean()
        new_array = coefficients*df.values+df.values
        new_df = pd.DataFrame(new_array, columns = df.columns)
        new_df[target_col] = mean_target_col
        df = df.append(new_df)
        df = df.sample(frac=1).reset_index(drop=True)
        logger.info(f"by adding a basic data augmentation we have improved by almost 2% the accuracy")
        return df

    
    def subsets(self): 
        ### 0. Baseline 1,2,3,4 all 4 classes
        df1,df2,df3,df4 = get_filtered_dfs(self.df)
        self.df_1_2_3_4 = pd.concat([df1, df2, df3, df4])
        self.df_1_2_3_4 = self.df_1_2_3_4.sample(frac=1).reset_index(drop=True)     # shuffle     

        ### 1. Entrenar con 1_2_3 vs 4.
        df1,df2,df3,df4 = get_filtered_dfs(self.df)  # reinitialize dfs
        df1.Efectividad = '1,2,3'
        df2.Efectividad = '1,2,3' # 1
        df3.Efectividad = '1,2,3' # 1
        self.df_1_2_3_vs_4 = pd.concat([df1, df2, df3, df4])
        self.df_1_2_3_vs_4 = self.df_1_2_3_vs_4.sample(frac=1).reset_index(drop=True)    # shuffle

        ### 2. Entrenar 2 vs 3_4
        df1, df2, df3, df4 = get_filtered_dfs(self.df)  # refresh dfs
        df3.Efectividad = '3,4'
        df4.Efectividad = '3,4'
        self.df_2_vs_3_4 = pd.concat([df2, df3, df4])
        self.df_2_vs_3_4 = self.df_2_vs_3_4.sample(frac=1).reset_index(drop=True)    # shuffle

        ### 3. 1vs4,  2vs4, 3vs4  (I.e, 4 vs all)
        df1, df2, df3, df4 = get_filtered_dfs(self.df)  # refresh dfs
        # self.df_1_4 = pd.concat([df1,df4])
        ### oversample df1!!! hardcoded to do this faster
        df1_oversampled = self.oversample(df1)
        self.df_1_4 = pd.concat([df1_oversampled,df4])
        self.df_2_4 = pd.concat([df2,df4])
        self.df_3_4 = pd.concat([df3,df4])
        self.df_1_4 = self.df_1_4.sample(frac=1).reset_index(drop=True)    # shuffle
        self.df_2_4 = self.df_2_4.sample(frac=1).reset_index(drop=True)    # shuffle
        self.df_3_4 = self.df_3_4.sample(frac=1).reset_index(drop=True)    # shuffle

        ### 4. 1vs all --> igual que el pto 3  -->  1vs2, 1vs3, 1vs4
        self.df_1_2 = pd.concat([df1,df2])
        self.df_1_3 = pd.concat([df1,df3])
        self.df_1_2 = self.df_1_2.sample(frac=1).reset_index(drop=True)    # shuffle
        self.df_1_3 = self.df_1_3.sample(frac=1).reset_index(drop=True)    # shuffle
        # df_1_4 = pd.concat([df3,df4])  # already done

        ### 5. Entrenar con 1_2 vs 3_4.
        df1,df2,df3,df4 = get_filtered_dfs(self.df)
        df1.Efectividad = '1,2'
        df2.Efectividad = '1,2'
        df3.Efectividad = '3,4'
        df4.Efectividad = '3,4'
        self.df_1_2_vs_3_4 = pd.concat([df1, df2, df3, df4])
        self.df_1_2_vs_3_4 = self.df_1_2_vs_3_4.sample(frac=1).reset_index(drop=True)    # shuffle
   
        self.cat_names = ['DIRECCIÓN:1 abierto;2 al cuerpo;3 a la T']
        # self.cont_names = ['V(km/h)', '[YA]', 'ZA', 'Znet', 'TIME', 'difV', '&(grados)', 'ANG. IN', 'dLinea']   
        self.cont_names = ['TIME', '[YA]', 'ZA', 'Znet', 'difV', 'ANG. IN', 'dLinea']  # 'TIME', 'V(km/h)',  ##### ****** here!!!!*****
        # self.cont_names = ['TIME', '[YA]', 'Znet', 'difV', '&(grados)', 'ANG. IN']  
        ### from correlation matriz: delete --> 'ZA' (keep ANG.IN), V(km/h) (keep TIME), & dLinea (keep GRADOS), 
        # then delete also ... DIRECCIÓN:1 abierto;2 al cuerpo;3 a la T, since it is a very bad variable
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


def train(df, final_epochs, metric, patience, cat_names, cont_names, y_names, seed=0):
    """metric: metric to monitor at last training"""
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
    
    dls = to.dataloaders(bs=1024)
    # learn = tabular_learner(dls, [256, 128, 128, 64], loss_func=FocalLossFlat(), metrics=[accuracy]) # loss_func=CrossEntropyLossFlat(),  256, 128, 128, 64
    # learn = tabular_learner(dls, [200,100], metrics=[accuracy, BalancedAccuracy(), Recall(), Precision(), APScoreMulti()]) # loss_func=CrossEntropyLossFlat(),  256, 128, 128, 64
    # learn = tabular_learner(dls, [256, 128, 128, 128, 64], loss_func=CrossEntropyLossFlat(), metrics=[accuracy, BalancedAccuracy()]) # loss_func=CrossEntropyLossFlat(),  256, 128, 128, 64
    learn = tabular_learner(dls, [256, 128, 128, 128, 64], loss_func=FocalLossFlat(), metrics=[accuracy, BalancedAccuracy()]) # loss_func=CrossEntropyLossFlat(),  256, 128, 128, 64
    
    ###
    # lr = 9e-3
    lr = learn.lr_find(show_plot=False).valley
    learn.lr = lr # lr/5
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


def main(df, data, hyperp, store_dir, split:list=[0], title='subsets df1 - df2', shap_subset='all', seed=0):
    """
    split has to be 0 or 1 values, where 0 means to calculate feat importance on training data,
        and 1 on Test Data
    """    
    print("split has to be 0 or 1 values, where 0 means to calculate feat importance on training data, ...")
    print("...and 1 on Test Data")
    print(f"{'#'*80}")
    store_dir = join(store_dir,data.ds_type)
    print(f"creating new dir: {store_dir}")
    os.makedirs(store_dir, exist_ok=True)
    base_file = os.path.split(store_dir)[1]
    final_epochs, metric, patience = hyperp.final_epochs, hyperp.metric, hyperp.patience
    learn, interp, splits, classif_report_dicto, confusion_matrix_np = train(df, final_epochs, metric, 
        patience, data.cat_names, data.cont_names, data.y_names, seed=seed)

    np.savetxt(join(store_dir, "confusion_matrix.csv"), np.around(confusion_matrix_np), delimiter=",", fmt="%d")   

    results = {'classif_report': classif_report_dicto}

    return learn


def train_123_vs_4(data, root_dir, ds_type, shap_subset='all', seed=0, plot_correlation=False):
    DS_deuce = '/home/javier/mis_proyectos/calculos_Fer/DATAJAVI_V5_deuce.csv'
    DS_advance = '/home/javier/mis_proyectos/calculos_Fer/DATAJAVI_V5_ad.csv'
    data = load_data(DS_deuce, DS_advance, ds_type=ds_type)
        
    learn = main(data.df_1_4, data, hyperp, join(root_dir, 'df_1_4'), split=[0, 1], 
                title='df_1_4', shap_subset=shap_subset, seed=seed)
    learn.save(f"model_1vs4_{ds_type}")
    return learn


def oversample(df):
    breakpoint()
    df2 = df.groupby(['Efectividad']).Efectividad.agg(['count'])
    df2['class %'] = round(df2['count']/len(df)*100, 1)
    # UNFINISHED!
    ### get which var has less samples
    
    ### augment by adding random gausian noise
    
    return df


def predict(learn, ds):
    """
    https://stackoverflow.com/questions/65561794/fastai-tabular-model-how-to-get-predictions-for-new-data
    
    model.get_preds is used get batch prediction on unseen data. You just need to apply the same 
    transformations on this new data as you did for training data.
    """
    learn.get_preds() # used get batch prediction on unseen data
    
    # dl = model.dls.test_dl(data.df_1_4, bs=64) # apply transforms
    dl = learn.dls.test_dl(data.df_1_4, bs=64) # apply transforms
    preds,  _ = learn.get_preds(dl=dl) # get prediction



if __name__=='__main__':
    import time
    t1 = time.time()
    class hyperp:
        final_epochs, metric, patience = 100, 'balanced_accuracy_score', 10 # 'balanced_accuracy_score', 3, # 'accuracy'
    seed = 1
    root_dir = '/home/javier/mis_proyectos/tennis_results/try2'

    DS_deuce = '/home/javier/mis_proyectos/calculos_Fer/DATAJAVI_V5_deuce.csv'
    DS_advance = '/home/javier/mis_proyectos/calculos_Fer/DATAJAVI_V5_ad.csv'
    shap_subset='all' # 10 # 'all'
    
    for ds_type in ['deuce', 'advance', 'both']:
        data = load_data(DS_deuce, DS_advance, ds_type=ds_type)
        learn = train_123_vs_4(data, root_dir, ds_type, shap_subset=shap_subset, seed=seed)
        print(f"\n\n\n{'#'*80}\n{'#'*80}\n{'#'*80}\n\n\n")
        print(len(learn.dls.valid_ds))
        print(len(learn.dls.train_ds))   
    
    
    t2 = time.time()
    print(f"The E2E process took {round((t2-t1)/60, 2)} minutes.")
    print("Bye")


"""
TODO: Make Report Again
We just improved the training. Now use the model to make predictions! and simulate results using new values!
"""
### analisis discriminante --> clustering?

# python train_v5_fastai.py 2>&1 | tee /home/javier/mis_proyectos/tennis_results/try2/train.log


