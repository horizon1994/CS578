import numpy as np
from sklearn import tree
import math
from math import sqrt
from sklearn.metrics import mean_squared_error


def decision_tree(X, y, depth):
	regr = tree.DecisionTreeRegressor(max_depth = depth, random_state = 34)
	regr.fit(X, y)
	return regr


def predict(regr, X):
	y_predicted = regr.predict(X)
	for each in range(y_predicted.shape[0]):
		y_predicted[each] = int(round(2*y_predicted[each]))/2
	return y_predicted


def get_RMSE(y_actual, y_predicted):
    return sqrt(mean_squared_error(y_actual, y_predicted))


def get_MAE(y_actual, y_predicted):
    return np.sum(np.absolute(y_actual.T - y_predicted))/y_actual.shape[0]


def get_spec_sens_prec_accu(y_actual, y_predicted):
    label = 0
    count = 0
    spec = 0
    sens = 0
    prec = 0
    accu = 0
    while label <= 5:
        tp, tn, fp, fn = 0, 0, 0, 0
        for each in range(y_actual.shape[0]):
            if y_actual[each] == label:
                if y_predicted[each] == label:
                    tp += 1
                else:
                    fn += 1
            else:
                if y_predicted[each] == label:
                    fp += 1
                else:
                    tn += 1
        if tp+fn == 0:  #there's no label 0 in test set
            sens += 1
        else:
            sens += tp/(tp+fn)  #sensitivity
        
        if tn+fp == 0:
            spec += 1
        else:
            spec += tn/(tn+fp)      #specificity
        
        if tp+fp == 0:
            prec += 1
        else:
            prec += tp/(tp+fp)      #precision
        
        accu += (tp+tn)/(tp+fp+tn+fn)  #accuracy
        label += 0.5
        count += 1
    return spec/count, sens/count, prec/count, accu/count


def kfoldcv(X, y, k, depth_array):
	opt_depth = 0
	opt_RMSE = float('inf')
	for depth in depth_array:
		n = X.shape[0]
		ave_RMSE = 0
		for i in range(1, k+1):
		    T = np.array([])  #T as test set
		    S = np.array([])  #S as train set
		    for j in range(math.floor(n*(i-1)/k), math.floor(n*i/k)):
		        T = np.append(T, [j])
		    for j in range(n):
		        S = np.append(S, [j])
		    S = np.setdiff1d(S, T)
		    T = T.astype(int)
		    S = S.astype(int)
		    #print(S)
		    regr = decision_tree(X[S, :], y[S, :], depth)
		    y_predicted = predict(regr, X[T, :])
		    ave_RMSE += get_RMSE(y[T, :], y_predicted)/k
		print('average RMSE of decision tree with max depth', depth, ':')
		print(ave_RMSE)
		if ave_RMSE < opt_RMSE:
			opt_RMSE = ave_RMSE
			opt_depth = depth
	return opt_depth

