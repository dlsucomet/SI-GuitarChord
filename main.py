import csv
import itertools
from sklearn.metrics import cohen_kappa_score, confusion_matrix, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from scipy.stats import uniform as sp_rand
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, Activation
from keras.models import Sequential

# Import the finalized, collated dataset
def import_dataset():
    with open('Datasets/finalized.csv', 'rt') as fin:
        cin = csv.reader(fin)
        data = [row for row in cin]
    return data


# Returns numpy compatible datasets that can be trained in models
def create_data(set):
    # Copy dataset
    workset = set[:]
    workset.remove(workset[0])

    # Take the targets and transform them to numerical points
    n_target = [row[44] for row in workset]
    target = []
    for item in n_target:
        if item == 'Down':
            target.append(0)
        elif item == 'Both':
            target.append(1)
        else:
            target.append(2)
    target = np.array(target)

    # Take the data lists
    n_data = np.array([row[0:-1] for row in workset])
    data = []
    for row in n_data:
        data.append([float(num) for num in row])
    data = np.array(data)

    target_names = np.array(['Down', 'Both', 'Up'])

    # Make a dictionary
    n_dict = {}
    n_dict['data'] = data
    n_dict['target'] = target
    n_dict['target_names'] = target_names

    return n_dict


def print_model_statistics(model, X_train, y_train, X_test, y_test):
    print('For model', model)
    print('Train set score: {:.4f}'.format(model.score(X_train, y_train)))
    print('Test set score: {:.4f}'.format(model.score(X_test, y_test)))
    kappa_scoring(model, X_test, y_test)


# Prints the kappa score and MAE for a model
def kappa_scoring(model, X_test, y_test):
    pred = model.predict(X_test)
    #print('Pred: {}'.format(pred))
    #print('Actl: {}'.format(y_test))
    cohen_score = cohen_kappa_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    print('Cohen kappa: {:.2f}'.format(cohen_score))
    print('Mean Absolute Error: {:.2f}'.format(mae))


# Prints the cross validation of three folds for a model
def cross_validation(model, dataset):
    score = cross_val_score(model, dataset['data'], dataset['target'], cv=10)
    print('Cross validation scores:', score)
    print('Average cross validation score: {:.2f}\n'.format(score.mean()))


# Returns the confusion matrix generated for a model
def confusion_scoring(model, X_test, y_test):
    pred = model.predict(X_test)
    confusion = confusion_matrix(y_test, pred)
    return confusion


# Plots a confusion matrix
# Set Normalize to True to enable normalization
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Main Routine
raw = import_dataset()
dataset = create_data(raw)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], random_state=0)

# Scaling for SVC and MLPs
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Make the models
# Tune the models before bagging/clustering into ensembles
# knn = KNeighborsClassifier(n_neighbors=5)
# bayes_gaussian = GaussianNB()
svm = SVC(kernel='rbf', gamma=0.0005, C=1000)
# mlp = MLPClassifier(activation='logistic', random_state=0, hidden_layer_sizes=[19])

# Create Ensembles and train
svm_ab = AdaBoostClassifier(svm, algorithm='SAMME')
svm_ab.fit(X_train_scaled, y_train)
#bayes_ab = GradientBoostingClassifier(init=bayes_gaussian).fit(X_train, y_train)
#knn_ab = RandomForestClassifier()
#mlp_ab = GradientBoostingClassifier(init=mlp).fit(X_train_scaled, y_train)


# Print the scores and kappa stats
print_model_statistics(svm_ab, X_train_scaled, y_train, X_test_scaled, y_test)
plot_confusion_matrix(confusion_scoring(svm_ab, X_test, y_test), classes=dataset['target_names'])
cross_validation(svm_ab, dataset)


'''
param_grid = {'gamma': sp_rand(), 'C': sp_rand()}
#model = SVC(kernel='rbf')
rsearch = RandomizedSearchCV(estimator=svm, param_distributions=param_grid, n_iter=100)
rsearch.fit(dataset['data'], dataset['target'])
print(rsearch)

print('{:.4f}'.format(rsearch.best_score_))
print('{:.4f}'.format(rsearch.best_estimator_.gamma))
print('{:.4f}'.format(rsearch.best_estimator_.C))
'''

