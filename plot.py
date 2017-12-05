import matplotlib.pyplot as plt
from KNN import *
from data_clean import *
from prank import *
from decision_tree import *


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

def plot(rsme):
    plt.title('ROC')
    x = [1.0, 0.9, 0.8, 0.7, 0.6]


    plt.xlim([0.6, 1.0])
    plt.ylim([0.9, 1.3])
    dim = np.arange(0.6,1.1,0.1)
    plt.ylabel('RSME')
    plt.xlabel('proportion of the orginal traing set')
    KNN, = plt.plot(x, rsme[0], 'b', label='KNN')
    #Prank, = plt.plot(x, rsme[1], 'r', label='Prank')
    #Tree, = plt.plot(x, rsme[2], 'g', label='Decision Tree')
    plt.xticks(dim)
    plt.legend(handles=[KNN])
    plt.grid()
    plt.show()

'''
for size in [0.0,0.1,0.2,0.3,0.4]:
    train, test = build_dict('ml-latest-small/ratings.csv', size)
    trained = trainKNN(train) # train on the whole training data
    prediction = get_prediction(500, train, test, trained) # generate predition for the testing data based on the trained model
    mae, rsme, spect, sent = get_MAE_RSME_spec_sen(prediction, test)
    print('on testing data, mae = %f, rsme = %f, spec = %f, sen = %f'%(mae, rsme, spect, sent))
    '''
plot([[1.114995, 1.115941, 1.102254, 1.122263, 1.102611]])
