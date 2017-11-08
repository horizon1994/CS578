from prank_data_clean import generate_matrix
from prank import *

def main():
	X_train, X_test, y_train, y_test = generate_matrix('./ml-latest-small/ratings.csv', './ml-latest-small/movies.csv', 0.2)
	# print(X_train)
	# print(X_test)
	# print(y_train)
	# print(y_test)

	L = 10  #iteration of PRank function
	print('result for L =', L)
	k = 11  #number of labels, which is fixed in our project
	theta, b = prank(L, k, X_train, y_train)
	print('theta:')
	print(theta)
	print('b:')
	print(b)
	
	mean_of_all = get_mean_of_all(X_train) #get mean rating for all movies, may be used in prediction
	y_predicted = predict(X_test, k, theta, b, mean_of_all)

	rmse = get_RMSE(y_test, y_predicted)
	print('rmse:')
	print(rmse)
	mae = get_MAE(y_test, y_predicted)
	print('mae:')
	print(mae)

	#specificity, sensitivity, precision, accuracy
	spec, sens, prec, accu = get_spec_sens_prec_accu(y_test, y_predicted)
	print('spec:')
	print(spec)
	print('sens:')
	print(sens)
	print('prec:')
	print(prec)
	print('accu:')
	print(accu)

main()

