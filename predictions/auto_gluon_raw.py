"""
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

###  Load and split data: 
df = pd.read_csv('df_2.csv')
df.drop(columns='Unnamed: 0', inplace=True)
# Creating a dataframe with 80%  # values of original dataframe
train_data = df.sample(frac = 0.8)
train_data.to_csv('train_data_gluon.csv')
# Creating dataframe with  # rest of the 20% values
test_data = df.drop(train_data.index)
train_data.to_csv('test_data_gluon.csv')

### instanciate predictor, then train the model
predictor = TabularPredictor(label='Efectividad').fit(train_data=train_data, verbosity = 2,presets="best_quality")
# predictions = predictor.predict(X_test)

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
with open("gluon_model.pkl", "wb") as f:
    pickle.dump(predictor, f)

predictor.save('gluon_model_saved_by_gluon')

print("\nthe model has been saved")

print("bye")

