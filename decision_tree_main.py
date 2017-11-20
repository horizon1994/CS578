from data_clean import generate_matrix
from decision_tree import *


def main():
	X_train, X_test, y_train, y_test = generate_matrix('./ml-latest-small/ratings.csv', './ml-latest-small/movies.csv', 0.2)

	#cross validation#
	depth_array = np.array([3, 5, 7, 9, 11])
	depth = kfoldcv(X_train, y_train, 5, depth_array)

	#prediction#
	print('predict using decision tree with max depth', depth, ':')
	regr = decision_tree(X_train, y_train, depth)
	y_predicted = predict(regr, X_test)

	rmse = get_RMSE(y_test, y_predicted)
	print('rmse:',rmse)
	mae = get_MAE(y_test, y_predicted)
	print('mae:', mae)

	#specificity, sensitivity, precision, accuracy
	spec, sens, prec, accu = get_spec_sens_prec_accu(y_test, y_predicted)
	print('spec:', spec)
	print('sens:', sens)
	print('prec:', prec)
	print('accu:', accu)

main()