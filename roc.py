import matplotlib.pyplot as plt
''' input example:
    sp = np.array([[0,1,0.7],[0, 0.8, 1],[1,0.5,0]])
    sen = np.array([[0,1, 0.7],[0, 0.9, 1],[1,0.6, 0]])
    sp[0], sp[1] and sp[2] are the specificity of KNN, Prank and Decision tree respectively
    sen[0], sen[1] and sen[2] are the sensitivity of KNN, Prank and Decision tree respectively'''

def plot(sp, sen):
    plt.title('ROC')
    KNN, = plt.plot(sp[0], sen[0], 'b', label='KNN')
    Prank, = plt.plot(sp[1], sen[1], 'r', label='Prank')
    Tree, = plt.plot(sp[2], sen[2], 'g', label='Decision Tree')
    plt.legend(handles=[KNN, Prank, Tree])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Sensitivity')
    plt.xlabel('Specificity')
    plt.show()