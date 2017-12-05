import matplotlib.pyplot as plt
from KNN import *
from data_clean import *
from prank import *
from decision_tree import *

'''
def build_dict(file, size):
    X_train, X_test, y_train, y_test = prank_data_split(file, 0.2)
    X_train_n, X_unused, y_train_n, y_unused = train_test_split(X_train, y_train, test_size=size, random_state=34)  # random_state
    train = X_train_n.join(y_train_n)
    test = X_test.join(y_test)
    ratings_train = train.groupby(train['userId'])
    ratings_test = test.groupby(test['userId'])
    dict_train, dict_test = {}, {}
    for name, group in ratings_train:
        for row in group.itertuples(index=True, name='Pandas'):
            try:
                dict_train[name].append((int(getattr(row, "movieId")), float(getattr(row, "rating"))));
            except KeyError:
                dict_train[name] = [(int(getattr(row, "movieId")), float(getattr(row, "rating")))];
    for name, group in ratings_test:
        for row in group.itertuples(index=True, name='Pandas'):
            try:
                dict_test[name].append((int(getattr(row, "movieId")), float(getattr(row, "rating"))));
            except KeyError:
                dict_test[name] = [(int(getattr(row, "movieId")), float(getattr(row, "rating")))];
    return dict_train, dict_test
'''
def plot(rmse):
    plt.title('RMSE vs proportion of the orginal training set')
    x = [1.0, 0.9, 0.8, 0.7, 0.6]


    plt.xlim([0.6, 1.0])
    plt.ylim([0.9, 1.7])
    dim = np.arange(0.6,1.1,0.1)
    plt.ylabel('RMSE')
    plt.xlabel('proportion of the orginal traing set')
    KNN, = plt.plot(x, rmse[0], 'b', label='KNN')
    Prank, = plt.plot(x, rmse[1], 'r', label='Prank')
    Tree, = plt.plot(x, rmse[2], 'g', label='Decision Tree')
    plt.xticks(dim)
    plt.legend(handles=[KNN, Prank, Tree])
    plt.grid()
    plt.show()

rmse_knn = [1.114995, 1.115941, 1.102254, 1.122263, 1.102611]
rmse_prank = [1.146553,1.448450,1.321510,1.121294,1.612194]
rmse_tree = [0.947170,0.948152,0.951396,0.956428,0.952985]
rmse = [rmse_knn, rmse_prank, rmse_tree]
plot(rmse)
