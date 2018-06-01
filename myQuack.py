'''

Scaffolding code for the Machine Learning assignment.

You should complete the provided functions and add more functions and classes as necessary.

Write a main function that calls the different functions to perform the required tasks
and repeat your experiments.


'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
from sklearn import preprocessing, decomposition

np.random.seed(10)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)

    '''
    return [(10144820, 'Petr', 'Ungar'), (1234568, 'Grace', 'Hopper')]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''
    Read a comma separated text file where
	- the first field is a ID number
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''
    dataframe = pd.read_csv(dataset_path, header=None)
    # Get the class label ('M' or 'B' and convert it into a numpy array
    y = np.array(dataframe.loc[:, 1])
    # Get the X data. We skip the first to columns (i.e., ID and class label)
    X = np.array(dataframe.loc[:, 2:])
    X.astype('float32')
    # Search where there are M and where B
    idx_M = np.argwhere(y == 'M')
    idx_B = np.argwhere(y == 'B')
    # Convert 'M' to 1 and 'B' to 0
    y[idx_M] = 1
    y[idx_B] = 0
    y = y.astype('int')  # Ask him about this line
    # Check shapes before returning
    if y.shape[0] != X.shape[0]:
        raise ValueError("Unequal number of columns between target (%d columns) and data vector (%d columns)." % (
        y.shape[0], X.shape[0]))

    return X, y


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NB_classifier(X_training, y_training):
    '''
    Build a Naive Bayes classifier based on the training set X_training, y_training.

    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''

    clf = GaussianNB()
    clf = clf.fit(X_training, y_training)
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DT_classifier(X_training, y_training, min_samples=1):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.
    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]
    @return
	clf : the classifier built in this function
    '''
    #tuning = tune_DT_classifier(DecisionTreeClassifier(), X_training, y_training)   #tuning more than one hyperpar improve accuracy
    clf = DecisionTreeClassifier(min_samples_leaf=min_samples)                      #tuning just one hyperparam
    clf.fit(X_training, y_training)
    return clf


#---------------------function to tune more hyperparms of the decision tree for more accuracy-----------------
def tune_DT_classifier(builder, X_train, y_train):
    '''  
    Find best params for a Decision Tree classifier based on the training set X_training, y_training.
    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]
    @return
	best params for decision tree
    '''
    params = {"criterion":["gini","entropy"],
              "splitter":["best","random"],
              "min_samples_split": [a for a in range(2,21,1)]}
    
    tree_cross_validation = RandomizedSearchCV(builder,params,cv=5)
    
    tree_cross_validation.fit(X_train, y_train)
        
    print("\n=========================Best params for DT:============================\n|",
          tree_cross_validation.best_params_,
          "|\n========================================================================")
    
    return tree_cross_validation.best_params_


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NN_classifier(X_training, y_training, neighbors=1):
    '''
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    clf = KNeighborsClassifier(n_neighbors=neighbors)
    clf = clf.fit(X_training, y_training)
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SVM_classifier(X_training, y_training, C_value=1):
    '''
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    clf = SVC(C=C_value, random_state=10)  # Set random seed for reproducibility
    clf = clf.fit(X_training, y_training)
    return clf


def cross_validate_model(builder, X_train, y_train, hyperparameter_list):
    """
    @ params
    X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]
    hyperparameter_list: List of hyperparameters to evaluate
    builder: Builder function to create model: build_SVM_classifier, build_NN_classifier, etc.

    @return
    Model
    """
    final_score = []
    models = []
    for hyperparameter in hyperparameter_list:
        clf = builder(X_train, y_train, hyperparameter)
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        mean_score = scores.mean()
        final_score.append(mean_score)
        models.append(clf)

    best_model = models[final_score.index(max(final_score))]
    best_hyperparameter = hyperparameter_list[final_score.index(max(final_score))]
    return best_model, best_hyperparameter, final_score


def test_model(model, X_test, y_test):
    """
    @params
    model: Model to test
    X_test: X_test[i,:] is the ith test case
    y_test: y_test[i] is the class label of X_test[i,:]

    @return:
    acc: Accuracy as (Number of Test cases - Number of misclassified)/Number of test cases
    """

    y_pred = model.predict(X_test)
    # Calculate accuracy as (Number of Test cases - Number of misclassified)/Number of test cases
    acc = (y_test.shape[0] - ((y_pred != y_test).sum())) / y_test.shape[0]
    return acc


def PCA_analysis(X_train, X_test):
    """
    @params
    X_training: X_training[i,:] is the ith example
    X_test: X_test[i,:] is the ith test case

    @return
    Data: Training data reduced to two dimensions
    test: Test data reduced to two dimensions
    pca:  Principal component analysis object
    """
    pca = decomposition.PCA(n_components=2)
    pca.fit(X_train)
    Data = pca.transform(X_train)
    test = pca.transform(X_test)
    return Data, test, pca


def visualize_PCA(Data, y_training):
    """
    Visualize PCA decomposition obtained with PCA_analysis() through a scatter plot
    Data: Training data reduced to two dimensions
    y_training: Labels for Data

    """
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.scatter(Data[:, 0], Data[:, 1])
    for i, label in enumerate(y_training):
        plt.annotate(label, (Data[i, 0], Data[i, 1]))
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
    plt.show()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    path = "C:/Users/BWH\Desktop/Clase/Australia/Artificial Intelligence/Machine learning/medical_records.data"
    X, y = prepare_dataset(dataset_path=path)
    # SHUFFLE THE DATA
    X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
    # Compute the mean and std for the training data. Apply to both training and test.
    scaler = preprocessing.StandardScaler().fit(X_training)
    X_training = scaler.transform(X_training)
    X_test = scaler.transform(X_test)
    # Naive Bayes model
    clf_nb = build_NB_classifier(X_training, y_training)
    # Evaluate hyperparameters
    print('Evaluating hyperparameters...\n')
    print('...Number of neighbors in KNN...\n')
    neighbors = [1, 2, 3, 10, 50, 100, 150, 200, 250, 300]
    clf_nn, best_neighbor, acc_per_neighbor = cross_validate_model(build_NN_classifier, X_training, y_training,
                                                                   neighbors)
    plt.plot(neighbors, acc_per_neighbor)
    plt.title('Accuracy per Neighbor', fontsize=20)
    plt.xlabel('Neighbors', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    print('Neighbours\t Accuracy\n', '\n'.join([' %d\t\t\t%f %%' % item for item in zip(neighbors, acc_per_neighbor)]))
    print('\n Best Neighbor is: %d' % best_neighbor)
    # C penalty value
    print('\n...C penalty value in SVM...\n')
    C_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    clf_svm, best_C_value, acc_per_C = cross_validate_model(build_SVM_classifier, X_training, y_training, C_list)
    plt.figure()
    plt.semilogx(C_list, acc_per_C)
    plt.title('Accuracy per C-value', fontsize=20)
    plt.xlabel('C-value', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    print('C value\t Accuracy\n', '\n'.join([' %1.3f\t\t%f %%' % item for item in zip(C_list, acc_per_C)]))
    print('\n Best C-value is: %d' % best_C_value)
    # Hyperparameter for decision tree
    # Hyperparameter for decision tree
    min_sample_list = list(np.arange(300) + 1)
    clf_dt, best_hyper, acc_per_hyper = cross_validate_model(build_DT_classifier, X_training,y_training, min_sample_list)
    print('Min_samples value\t Accuracy\n', '\n'.join([' %1.3f\t\t%f %%' % item for item in zip(min_sample_list, acc_per_hyper)]))
    print("Best min_samples_leaf value is:", best_hyper)

    print('\nTest stage\n')
    print('...Processing...\n')
    # Test
    acc_nb = test_model(clf_nb, X_test, y_test)
    acc_knn = test_model(clf_nn, X_test, y_test)
    acc_svm = test_model(clf_svm, X_test, y_test)
    acc_dt = test_model(clf_dt, X_test, y_test)
    print('Naïve Bayes model achieved: %1.3f %% accuracy in the test set' % acc_nb)
    print('KNN model achieved: %1.3f %% accuracy in the test set' % acc_knn)
    print('SVM model achieved: %1.3f %% accuracy in the test set' % acc_svm)
    print('Decission tree achieved: %1.3f %% accuracy in the test set' % acc_dt)

    plt.show()
