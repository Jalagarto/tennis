from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from sklearn.metrics import classification_report

predictor = TabularPredictor.load('final')

print(f"\n\n training report: ")
predictor.fit_summary()

print(f"\n\n validation leaderboard: ")
predictor.leaderboard()

test_data = pd.read_csv('../test_data_gluon.csv')

print("\nEvaluation_with sklearn:")
y_pred = predictor.predict(test_data, model='NeuralNetTorch_DSTL')  # 'LightGBM_BAG_L1')
print(classification_report(test_data.Efectividad.values, y_pred))

print(f"\n\n feature_importance: ")
print(predictor.feature_importance(data='test_data_gluon.csv'))

"""
          model   score_val  pred_time_val
LightGBM_BAG_L1   0.885844       0.329239
"""
