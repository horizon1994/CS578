
import numpy as np
import csv
#import random
import math
import pandas as pd


def prank_data_clean(file_ratings, file_movies):
	ratings = pd.read_csv(file_ratings)
	movies = pd.read_csv(file_movies)

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
	for column in final.ix[:,7:-2]:
    	curr_genres = final[final[column]==1.0].groupby("userId").rating_x.mean()
    	user_genres = pd.concat([user_genres, curr_genres], axis=1)
	user_genres = user_genres.fillna(0)

	'''compute ng: number of genres for each movie'''
	genres = final.ix[:,7:-2]
	sum_genres = genres.sum(axis=1)
	number_genres = pd.DataFrame({'number_genres':sum_genres.values})
	final = final.join(number_genres)

	'''compute mg*ug/ng, mg is the vector of boolean values for all genres'''
	result = []
	for each in final.index:
		line = final.iloc[each]
		userId = line.userId
		ug = user_genres.iloc[userId - 1]
		mg = final.ix[:,7:-3].iloc[each]
		ng = final.ix[:,-1].iloc[each]
		if ng == 0:
			result.append(0)
		else:
			result.append(np.dot(ug,mg)/ng)
	result = pd.DataFrame({'mg*ug/ng':result})
	final = final.join(result)



