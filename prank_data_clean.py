import numpy as np
import csv
import math
import pandas as pd
from sklearn.model_selection import train_test_split


def prank_data_split(file_ratings, size):
    ratings = pd.read_csv(file_ratings)
    X = ratings[['userId', 'movieId']]
    y = ratings['rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 34) #random_state
    return X_train, X_test, y_train, y_test


def train_data_clean(X_train, y_train, file_movies):
    #ratings = pd.read_csv(file_ratings)
    movies = pd.read_csv(file_movies)
    ratings = X_train.join(y_train)

    '''extract genres of movies'''
    genres = movies.genres.str.split('|', expand = True).stack()
    genres = pd.get_dummies(genres, prefix = 'g').groupby(level = 0).sum()
    movie_genres = movies.join(genres, how = 'outer')

    '''compute average ratings of movies'''
    movie_average = ratings['rating'].groupby(ratings['movieId']).mean()
    movie_genres_ave = movie_genres.join(movie_average, on = 'movieId')
    overall = pd.merge(ratings, movie_genres_ave, on = "movieId")
    overall.sort_values('userId')

    '''compute average ratings of users'''
    user_average = overall.groupby("userId").rating_x.mean()
    user_ave = pd.DataFrame({'userId':user_average.index, 'user_average':user_average.values})
    final = overall.merge(user_ave, on = "userId")

    '''compute ug: average user ratings for each genre'''
    user_genres = pd.DataFrame()
    for column in final.ix[:,6:-2]:
    	curr_genres = final[final[column]==1.0].groupby("userId").rating_x.mean()
    	user_genres = pd.concat([user_genres, curr_genres], axis=1)
    	user_genres = user_genres.fillna(0)

    '''compute ng: number of genres for each movie'''
    genres = final.ix[:,6:-2]
    sum_genres = genres.sum(axis=1)
    number_genres = pd.DataFrame({'number_genres':sum_genres.values})
    final = final.join(number_genres)

    '''compute mg*ug/ng, mg is the vector of boolean values for all genres'''
    result = []
    for each in final.index:
    	line = final.iloc[each]
    	userId = line.userId
    	ug = user_genres.iloc[userId - 1]
    	mg = final.ix[:,6:-3].iloc[each]
    	ng = final.ix[:,-1].iloc[each]
    	if ng == 0:
    		result.append(0)
    	else:
    		result.append(np.dot(ug,mg)/ng)
    result = pd.DataFrame({'mg*ug/ng':result})
    final = final.join(result)

    return final, user_genres, movie_genres


def test_data_clean(X_test, final, user_genres, movie_genres):
	'''get genres of movies in X_test'''
	test_genres = pd.merge(X_test, movie_genres, on = 'movieId')

	'''find corresponding movie average from training set'''
	movie_average = []
	for each in test_genres.movieId:
	    num = final[final['movieId'] == each]['rating_y'].unique()
	    if num:
	        num = num[0]
	    else:
	        num = 0
	    movie_average.append(num)

	'''find corresponding user average from training set'''
	user_average = []
	for each in test_genres.userId:
	    num = final[final['userId'] == each]['user_average'].unique()
	    if num:
	        num = num[0]
	    else:
	        num = 0
	    user_average.append(num)

	test_genres = test_genres.join(pd.DataFrame({'rating_y':movie_average}))
	test_genres = test_genres.join(pd.DataFrame({'user_average':user_average}))

	'''compute ng: number of genres for each movie'''
	genres = test_genres.ix[:,6:-2]
	sum_genres = genres.sum(axis=1)
	number_genres = pd.DataFrame({'number_genres':sum_genres.values})
	test_genres = test_genres.join(number_genres)

	'''compute mg*ug/ng, mg is the vector of boolean values for all genres'''
	result = []
	for each in test_genres.index:
	    line = test_genres.iloc[each]
	    userId = line.userId
	    ug = user_genres.iloc[userId - 1]
	    mg = test_genres.ix[:,6:-3].iloc[each]
	    ng = test_genres.ix[:,-1].iloc[each]
	    if ng == 0:
	        result.append(0)
	    else:
	        result.append(np.dot(ug,mg)/ng)
	result = pd.DataFrame({'mg*ug/ng':result})
	test_genres = test_genres.join(result)

	return test_genres


def generate_matrix(file_ratings, file_movies, size):
	X_train, X_test, y_train, y_test = prank_data_split(file_ratings, size)

	'''clean training data'''
	final, user_genres, movie_genres = train_data_clean(X_train, y_train, file_movies)
	X = final.ix[:,6:]
	X = X.drop('number_genres', axis = 1)
	X_train = X.as_matrix()
	y = final.ix[:,2:3]
	y_train = y.as_matrix()

	'''clean test data'''
	test_genres = test_data_clean(X_test.join(y_test), final, user_genres, movie_genres)
	X = test_genres.ix[:,6:]
	X = X.drop('number_genres', axis = 1)
	X_test = X.as_matrix()
	y = test_genres.ix[:,2:3]
	y_test = y.as_matrix()

	return X_train, X_test, y_train, y_test




