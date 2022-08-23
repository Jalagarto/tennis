"""
finally let's map features:
	1. get mean values and play with all features --> predict for different values and get
	a window of values that make the shots effective.

	2. We will get a range of data for 20%-80% of feat var, then iterate over them,
	but this time we get a mesh of values

	3. make the app for them to play
"""

from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from sklearn.metrics import classification_report
import time
import numpy as np
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
# pd.options.display.max_columns = None

predictor = TabularPredictor.load('./AutogluonModels/final')

# print(f"\n\n training report: ")
# predictor.fit_summary()
# 
print(f"\n\n validation leaderboard: ")
# predictor.leaderboard()

# test_data = pd.read_csv('./AutogluonModels/test_data_gluon.csv')
test_data = TabularDataset('test_data_gluon.csv')
test_data.drop(columns='Unnamed: 0', inplace=True)

train_data = TabularDataset('test_data_gluon.csv')
train_data.drop(columns='Unnamed: 0', inplace=True)

# y_pred = predictor.predict(test_data)

# print("\nEvaluation_with sklearn:")
# t = time.process_time()
# y_pred = predictor.predict(test_data, model='LightGBM_BAG_L1')
# cls_report = classification_report(test_data.Efectividad.values, y_pred, output_dict=True)
# elapsed_time = time.process_time() - t
# print(cls_report)

test_acc = predictor.evaluate(test_data, model='NeuralNetTorch_DSTL')['accuracy']
print('\ntest_accuracy: ', test_acc)

y_pred = predictor.predict_proba(test_data, model='NeuralNetTorch_DSTL', as_pandas=True, as_multiclass=True)

print("\ny_pred: \n", y_pred)

# predictor.distill()  # this achieved a better result --> new model: 'NeuralNetTorch_DSTL'
# ... which is more powerful and much faster  





def get_df_map(df_mean):
	"""
	get_map_and_infer
	"""
	### hardcoded_values: 
	dicto = {
		'serve_class': ['NOT RETURNED', 'RETURNED', 'FAULT', 'ACE'],
		'shotCount':[1,2,4,8,12],
		'1º o 2º saque':[0,1],
		'gameNumber' :[1,3,6,8,12],
		'Lado(1:Iguales;0:Ventaja)' :[0,1],
		'pointNumber':[1,2,4,5,10],
		'PostVx' :[-49, -20, 8, 22, 45],
		'Z1 (h)' :[1.5, 2.7, 2.8, 2.9, 4.4],
		'V (m/s)' :[21, 42, 46, 51, 64],
		'timeHIT' :[0,0.65,1,2,9],
		'PreVz' :[-10.5, -8.2,-7.8,-7.4,-2.8]
		}

	new_list_dict = []
	for k in dicto.keys():
		for v in dicto[k]:
			dicto_q50 = df_mean.to_dict()
			dicto_q50['serve_class'] = 'RETURNED'
			dicto_q50[k]=v
			new_list_dict.append(dicto_q50)

	final_data_grid = pd.DataFrame(new_list_dict)

	y_pred = predictor.predict_proba(final_data_grid, model='NeuralNetTorch_DSTL', as_pandas=True, as_multiclass=True)
	# print('final_data_grid: ', y_pred)

	final_data_grid['Efectividad'] = y_pred[1]
	final_data_grid.to_csv('grid_1.csv')





def get_simple_window(train_data, test_data):
	"""
	get a dicto with max-min values.
	What if the values have some discontinuities?
	show max min or even show a map
	"""
	# print(test_df.values)
	feature_generator = AutoMLPipelineFeatureGenerator()
	X_train = train_data.drop(labels=['Efectividad'], axis=1)
	y_train = train_data['Efectividad']
	X_train_transformed = feature_generator.fit_transform(X=X_train, y=y_train)
	X_test_transformed = feature_generator.transform(test_data)
	print('\nX_test_transformed: \n', list(X_test_transformed.columns))
	d = X_test_transformed.describe()
	print("\ndescribe rows: ", list(d.index))
	return d.loc[['min', '25%','50%','75%', 'max']]
	# get_df_map(d.loc['50%'])

	### predict 10 times for each variable

	### draw a heatmap
	

if __name__ == '__main__':
	get_simple_window(train_data, test_data)

"""
Main features (feature importance):
-----------------------------------
											importance stddev    p_value   n  p99_high   p99_low
											---------- -------   -------   -  --------   --------
shotCount                                   0.161000   0.017692  0.002000  3  0.262376  0.059624
serve_class                                 0.112333   0.008145  0.000874  3  0.159002  0.065664
1º o 2º saque                               0.026667   0.010116  0.022386  3  0.084632 -0.031299
gameNumber                                  0.005000   0.002646  0.041007  3  0.020160 -0.010160                                     
Lado(1:Iguales;0:Ventaja)                   0.002667   0.000577  0.007634  3  0.005975 -0.000642
PostVx                                      0.001000   0.001000  0.112702  3  0.006730 -0.004730
pointNumber                                 0.001000   0.001000  0.112702  3  0.006730 -0.004730
Z1 (h)                                      0.001000   0.001000  0.112702  3  0.006730 -0.004730
V (m/s)                                     0.000667   0.001528  0.264298  3  0.009420 -0.008086
PreVz                                       0.000667   0.000577  0.091752  3  0.003975 -0.002642
timeHIT                                     0.000667   0.000577  0.091752  3  0.003975 -0.002642
"""
