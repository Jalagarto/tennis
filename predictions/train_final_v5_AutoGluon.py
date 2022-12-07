"""
copy of auto_gluon_raw.py
1. split data,
2. train model, 
3. summarize results and calculate feature importance
4. Evaluate
5. Save the model
"""

import pickle
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import classification_report, confusion_matrix


# DS_ = input('dataset: ')
# DS = f'./final_ds/{DS_}.csv'
# eval_metric = input("choose between: [‘accuracy’, ‘balanced_accuracy’, ‘f1’, ‘f1_weighted’, etc']: ") or 'accuracy'

def main(DS, eval_metric='balanced_accuracy', presets='best_quality'):
    ###  Load and split data: 
    df = pd.read_csv(DS)

    ### try different DS setups
    df1 = df[df.Efectividad==1]
    df2 = df[df.Efectividad==2]
    df3 = df[df.Efectividad==3]
    df4 = df[df.Efectividad==4]

    df_1_4 = pd.concat([df1,df4])  # dataset with 1 and 4s
    df_2_3 = pd.concat([df2,df3])  # dataset with 2 & 3s

    df2.Efectividad = 1
    df3.Efectividad = 4
    df_Efficient_vs_notEfficient = pd.concat([df1, df2, df3, df4])  # all together --> Effective vs not Effective    
    df = df_1_4 # df_Efficient_vs_notEfficient

    # Creating a dataframe with 80%  # values of original dataframe
    train_data = df.sample(frac = 0.8)
    train_data.to_csv(f'train_data_gluon.csv')
    # Creating dataframe with  # rest of the 20% values
    test_data = df.drop(train_data.index)
    test_data.to_csv(f'test_data_gluon.csv')

    ### instanciate predictor, then train the model
    predictor = TabularPredictor(label='Efectividad', eval_metric=eval_metric).fit(train_data=train_data,
                        verbosity = 2,
                        presets=presets) # "interpretable") # "best_quality")
    # predictions = predictor.predict(X_test)

    best_model_name = predictor._get_model_best()

    if presets=='interpretable':
        predictor.print_interpretable_rules() # can optionally specify a model name or complexity threshold
    else:
        cls, columns, labels = predictor.explain_classification_errors(test_data, model=best_model_name) # , model='NeuralNetTorch_BAG_L1')
        ### Explain classification errors by fitting a rule-based model to them
        ### /home/javier/pyenvs/tennis/lib/python3.8/site-packages/autogluon/tabular/predictor/predictor.py

    ### summary
    # print("\n:fit_summary") 
    # print(predictor.fit_summary())
    print("\n:leaderboard")
    print(predictor.leaderboard(test_data, silent=True))

    ### calculate feature importance
    print("\n:feature_importance")
    print(predictor.feature_importance(data=test_data))

    ### evaluate:
    print("\nEvaluation:")
    print(predictor.evaluate(test_data))

    print(f"\nEvaluation_with sklearn. Model: {best_model_name}")
    y_pred = predictor.predict(test_data, model=best_model_name)
    
    print(classification_report(test_data.Efectividad.values, y_pred))

    DS_short = DS.rsplit('/', 1)[1].rsplit('.', 1)[0]

    ### save the model
    with open(f"gluon_model{DS_short}.pkl", "wb") as f:
        pickle.dump(predictor, f)

    predictor.save(f'gluon_model_saved_by_gluon')

    print("\nThe model has been saved")

    print("bye")


if __name__=='__main__':
    # eval_metric = 'input('"choose between: [‘accuracy’, ‘balanced_accuracy’, ‘f1’, ‘f1_weighted’, etc']: ") or 'accuracy'    
    DS = '/home/javier/mis_proyectos/calculos_Fer/DATAJAVI_V5_deuce.csv'
    eval_metric, presets = 'recall_weighted', 'best_quality'    # "interpretable") # "best_quality")
    main(DS, eval_metric, presets)   # eval_metric, presets are optional
