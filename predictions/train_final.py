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
    ### Load dataset
    print('Dataset Name: ', DS, '\n')
    DS_short = DS
    DS = f'./final_ds/{DS}.csv'

    ###  Load and split data: 
    df = pd.read_csv(DS)
    df.drop(columns='Unnamed: 0', inplace=True)
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

    if presets=='interpretable':
        predictor.print_interpretable_rules() # can optionally specify a model name or complexity threshold

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

    print("\nEvaluation_with sklearn:")
    y_pred = predictor.predict(test_data)

    print(classification_report(test_data.Efectividad.values, y_pred))

    ### save the model
    with open(f"gluon_model{DS_short}.pkl", "wb") as f:
        pickle.dump(predictor, f)

    predictor.save(f'gluon_model_saved_by_gluon')

    print("\nthe model has been saved")

    print("bye")


if __name__=='__main__':
    # eval_metric = 'input('"choose between: [‘accuracy’, ‘balanced_accuracy’, ‘f1’, ‘f1_weighted’, etc']: ") or 'accuracy'    
    DS, eval_metric, presets = 'df_3', 'recall_weighted', 'interpretable'    # "interpretable") # "best_quality")
    main(DS, eval_metric, presets)   # eval_metric, presets are optional
