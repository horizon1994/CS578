import converter as cv
import numpy as np
from operator import itemgetter
import math
import random


''' given two sorted lists ui and uj, return the common elements of these two lists'''
def get_common(ui, uj):
    common = []
    i = j = 0
    while i < len(ui) and j < len(uj):
        if ui[i][0] == uj[j][0]:
            common.append((ui[i][0],ui[i][1], uj[j][1]));
        if ui[i][0] <= uj[j][0]:
            i += 1;
        else:
            j += 1;
    return common;

''' return the sum of the ratings from a list of tuples where each tuple has form: (uid, rating given by uid).'''
def get_sum(ui):
    sum = 0;
    for x in ui:
        sum += x[1];
    return sum;

''' return the average rating given by user ui'''
def get_mean(ui):
    return get_sum(ui)/len(ui);


''' given a floating point number, return the valid rating that is closest to it'''
def get_valid_rating(x):
    lower_bound = math.floor(x);
    upper_bound = math.ceil(x);
    median = lower_bound + 0.5;
    min = 1;
    ret = -1;
    for y in [lower_bound, upper_bound, median]:
        if abs(x-y) < min:
            min = abs(x-y);
            ret = y;
        if abs(x-y) == min:
            if ret != -1:
                ret = random.choice([ret, y]);
            else:
                ret == y;
    return ret;





''' return the rescaled Pearson Correlation Correlation between user i and user j'''
def get_pearson(i,j, dict):
    numerator = 0.0;
    vi = 0.0;
    vj = 0.0;
    commonMovie = get_common(dict[i], dict[j]);
    M = len(commonMovie);
    if M>0:
        mui = get_mean(dict[i]);
        muj = get_mean(dict[j]);
        for x in commonMovie:
            numerator += (x[1]-mui)*(x[2]-muj);
            vi += math.pow(x[1] - mui, 2);
            vj += math.pow(x[2] - muj, 2);
        denominator = math.sqrt(vi*vj)/M;
        if denominator==0:
            return 0.0;
        else:
            return ((numerator/M)/denominator+1)/2;
    return 0.0;

''' for each user in the training set, calculate the Pearson Correlation Coefficient between this user and all the other users,
    and store them in a tuple that has form (j, p_(i,j))'''
def trainKNN(train):
    dict = {};
    for i in train.keys():
        temp = [];
        for j in train.keys():
            if i!=j:
                temp.append((j, get_pearson(i,j,train)));
        dict[i] = temp;
    return dict;

''' find the rating of movie K in a user's rating list.
    If it is found, return the rating
    Otherwise, return None'''
def findK(k, left, right, list):
    pivot = math.floor((left+right)/2);
    if left>right:
        return None;
    if k==list[pivot][0]:
        return list[pivot][1];
    if k<list[pivot][0]:
        return findK(k, left, pivot-1, list);
    if k>list[pivot][0]:
        return findK(k, pivot+1, right, list);


''' predict the user uId's rating of movie mId.
    find k users that are most similar to user uId and has rated this moive
    If none are found, return the average of the rating given by user uId
    Otherwise, predict the rating based on the weighted average of the ratings given by those users that are most similar to user uId'''
def predict(k, uId, mId, trained, dict):
    list = [];
    temp = sorted(trained[uId], key=itemgetter(1), reverse=True);
    muk = 0.0;
    print(temp);
    for x in temp[0:k+1]:
        rating = findK(mId, 0,len(dict[x[0]]), dict[x[0]]);
        if rating!=None:
            list.append((x[0],rating));
            muk+=rating;
    if len(list)>0:
        muk = muk/len(list);
        numerator = 0.0;
        denominator = 0.0;
        for x in list:
            if x[0]<uId:
                numerator += trained[uId][x[0]-1][1]*(x[1]-muk);
                denominator += abs(trained[uId][x[0]-1][1]);
            else:
                numerator += trained[uId][x[0] - 2][1] * (x[1] - muk);
                denominator += abs(trained[uId][x[0] - 2][1]);
        return get_valid_rating(numerator/denominator + muk);
    else:
        print('no list');
        return get_valid_rating(get_mean(dict[uId]));
