import matplotlib.pyplot as plt
import KNN
from data_clean import *
import prank
import decision_tree
'''
def get_list():
    train, test = build_dict('../dataset/ratings.csv', 0.2)
    X_train, X_test, y_train, y_test = prank_data_split('../dataset/ratings.csv', 0.2)
    X_train, X_test, y_train, y_test = generate_matrix(X_train, X_test, y_train, y_test, '../dataset/movies.csv')
    sp = [[],[],[]]
    sen = [[],[],[]]
    for k in [2, 20, 40, 80, 160, 320]:
        trained = KNN.trainKNN(train) # train on the whole training data
        prediction = KNN.get_prediction(k, train, test, trained) # generate predition for the testing data based on the trained model
        mae, rsme, spect, sent = KNN.get_MAE_RSME_spec_sen(prediction, test)
        sp[0].append(spect)
        sen[0].append(sent)
    for L in [1,5,10,50,100,200]:
        k = 11  # number of labels, which is fixed in our project
        theta, b = prank.prank(L, k, X_train, y_train)
        mean_of_all = prank.get_mean_of_all(X_train)  # get mean rating for all movies, may be used in prediction
        y_predicted = prank.predict(X_test, k, theta, b, mean_of_all)
        spec, sens, prec, accu = prank.get_spec_sens_prec_accu(y_test, y_predicted)
        sp[1].append(spec)
        sen[1].append(sens)
    for depth in range(1,12):
        regr = decision_tree.decision_tree(X_train, y_train, depth)
        y_predicted = decision_tree.predict(regr, X_test)
        # specificity, sensitivity, precision, accuracy
        spec, sens, prec, accu = decision_tree.get_spec_sens_prec_accu(y_test, y_predicted)
        sp[2].append(spec)
        sen[2].append(sens)
    return sp, sen
    '''

def plot(sp, sen):
    plt.title('ROC')
    for i in range(len(sp)):
        sp[i] = list(map(lambda x:1-x, sp[i]))
        sen[i].extend((0,1))
        sp[i].extend((0,1))
        sen[i], sp[i] = zip(*sorted(zip(sen[i], sp[i])))
    KNN, = plt.plot(sp[0], sen[0], 'b', label='KNN')
    Prank, = plt.plot(sp[1], sen[1], 'r', label='Prank')
    Tree, = plt.plot(sp[2], sen[2], 'g', label='Decision Tree')
    plt.legend(handles=[KNN, Prank, Tree])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Sensitivity')
    plt.xlabel('1-Specificity')
    plt.show()

sp_knn = [0.914715,0.915061,0.915044,0.915005,0.914611,0.913886]
sen_knn = [0.216077,0.218079,0.218346,0.218432,0.216744,0.212653]
sp_prank = [0.909254,0.908832,0.909238,0.911202,0.913902,0.912513]
sen_prank = [0.198710,0.198757,0.206928,0.215047,0.226520,0.220575]
sp_tree = [0.913977,0.913702,0.914700,0.917030,0.918300,0.917735,0.917499,0.917469,0.917676,0.917667,0.917810]
sen_tree = [0.197038,0.218240,0.233431,0.225568,0.243366,0.239331,0.240045,0.242531,0.243016,0.242281,0.246050]
plot([sp_knn, sp_prank, sp_tree],[sen_knn, sen_prank, sen_tree])
