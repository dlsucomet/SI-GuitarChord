import csv
import itertools
from sklearn.metrics import cohen_kappa_score, confusion_matrix, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from scipy.stats import uniform as sp_rand
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BoostingWrapper:
    def __init__(self, est):
        self.est = est

    def predict(self, X):
        return self.est.predict_proba(X)[:, 1]

    def fit(self, X, y):
        self.est.fit(X, y)


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
    # Print testing scores
    print('For model', model)
    print('Train set score: {:.4f}'.format(model.score(X_train, y_train)))
    print('Test set score: {:.4f}'.format(model.score(X_test, y_test)))

    # Create a prediction and compute value scores
    pred = model.predict(X_test)
    cohen_score = cohen_kappa_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    print('Cohen kappa: {:.2f}'.format(cohen_score))
    print('Mean Absolute Error: {:.2f}'.format(mae))


# Prints the cross validation of ten folds for a model
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


# Random Optimization for Tuning
def optimize_random(model, params, iterations, dataset):
    print('Ranomized Optimization Results:')
    param_grid = {param: sp_rand() for param in params}
    rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=iterations)
    rsearch.fit(dataset['data'], dataset['target'])
    #print(rsearch)

    print('Best score achieved: {:.4f}'.format(rsearch.best_score_))
    #for param in rsearch.best_estimator_:
    #    print('{}: {:.4f}'.format(param, rsearch.best_estimator_.param))
    print(rsearch.best_estimator_)
    print('\n')


# Main Routine
raw = import_dataset()
dataset = create_data(raw)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], random_state=0)

# Scaling for SVC and MLP
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(dataset['data'])


# Make the models
pipe = Pipeline([
    ('scaler', MinMaxScaler()),
    ('svm', SVC(C=1000, gamma=0.0005)),
    ('cross_val', cross_val_score())
])
pipe.fit(X_train, y_train)
print_model_statistics(pipe, X_train_scaled, y_train, X_test_scaled, y_test)


svm = SVC(C=1000, gamma=0.0005)
svm.fit(X_train_scaled, y_train)
svm_ab = AdaBoostClassifier(base_estimator=svm, algorithm='SAMME').fit(X_train_scaled, y_train)
print_model_statistics(svm_ab, X_train_scaled, y_train, X_test_scaled, y_test)

cvs = cross_val_score(svm_ab, X_scaled, dataset['target'], cv=10)
print('Cross validation scores:', cvs)
print('Average cross validation score: {:.2f}\n'.format(cvs.mean()))

mlp = MLPClassifier(hidden_layer_sizes=[3], activation='logistic').fit(X_train_scaled, y_train)
print_model_statistics(mlp, X_train_scaled, y_train, X_test_scaled, y_test)


'''
grid = [
    {'svm__C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
     'svm__gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
]

gscv = GridSearchCV(pipe, param_grid=grid, cv=10).fit(X_train_scaled, y_train)
print("Best parameters: {}".format(gscv.best_params_))
print("Best cross-validation score: {:.2f}".format(gscv.best_score_))

# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
dataframe = pd.DataFrame(X_train, columns=dataset['target_names'])
# create a scatter matrix from the dataframe, color by y_train
grr = pd.scatter_matrix(dataframe, c=y_train, figsize=(15, 15), marker='o',
                            hist_kwds={'bins': 20}, s=60, alpha=.8)
'''
