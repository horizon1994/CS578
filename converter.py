import numpy as np
import csv
import random
import math



''' read ratings from ratings file and store them to dict that use userIDs as keys.'''
def rating2dict(file):
    urdict = {};
    with open(file, 'r') as f:
        reader = csv.reader(f);
        next(reader);
        for row in reader:
            try:
                urdict[row[0]].append((row[1], float(row[2])));
            except KeyError:
                urdict[row[0]] = [(row[1], float(row[2]))];
        for x in urdict.keys(): # shuffle the ratings by each user
            random.shuffle(urdict[x]);
    return urdict;

''' split the dict into trainig and testing. '''
def dict_train_test_split(dict, i, k):
    train = {};
    test = {};
    for x in dict.keys():
        n = len(dict[x]);
        count = 0;
        index = math.floor(n*(i-1)/k);
        bound = math.floor(n*i/k)-index;
        test[x] = [0]*bound;
        while count < bound:
            test[x][count] = dict[x].pop(index);
            count+=1;
        train[x] = dict[x];
    return train, test;

''' read movie descriptions from movies file and store them into dict that uses movieIDs as keys. '''
def movie2dict(file):
    mdict = {};
    with open(file, 'r') as f:
        reader = csv.reader(f);
        next(reader);
        for row in reader:
            mdict[row[0]]= tuple(row[2].split('|'));
    return mdict;










