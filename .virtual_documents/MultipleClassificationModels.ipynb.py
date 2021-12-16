import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


titanic_df = pd.read_csv("./datasets/titanic_mine.csv")
titanic_df.head()


FEATURES = list(titanic_df.columns[1:])
FEATURES


# dict to hold score of each model on train and test dataset
result_dict = {}


# helper function that ret a dict of metrics score based on prediction and actual target passed
def summarize_classification(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred, normalize = True)   # since norm = True, acc in term of fraction
    num_acc = accuracy_score(y_test, y_pred, normalize = False)
    
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    return {"accuracy": acc,
           "precision": prec,
           "recall": recall,
           "accuracy_count": num_acc}


# another helper fun to build the model, classfier_fn is classifier fn created by me
def build_model(classifier_fn, name_of_y_col, name_of_x_cols, dataset, test_frac=0.1):
    X = dataset[name_of_x_cols]
    Y = dataset[name_of_y_col]
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = test_frac)
    model = classifier_fn(x_train, y_train)
    
    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    
    train_sumary = summarize_classification(y_train, y_pred_train)
    test_summary = summarize_classification(y_test, y_pred)
    
    pred_results = pd.DataFrame({"y_test": y_test,
                                "y_pred": y_pred})
    
    model_crosstab = pd.crosstab(pred_results.y_pred, pred_results.y_test)
    return {
        "training": train_sumary,
        "test": test_summary,
        "confusion_matrix": model_crosstab
    }
    

def compare_results():
    """
    fun compares result of the diff clf models built
    """
    for key in result_dict:
        print("Classification: ", key)
        
        print()
        print("Training data")
        # gets info about model on training data
        for score in result_dict[key]["training"]:
            print(score, result_dict[key]["training"][score])
            
        print()
        print("Test data")
        for score in result_dict[key]["test"]:
            print(score, result_dict[key]["test"][score])
            
        print()


# log reg fn
def logistic_fn(x_train, y_train):
    model = LogisticRegression(solver="liblinear")
    model.fit(x_train, y_train)
    
    return model


result_dict["survived - logistic"] = build_model(logistic_fn, "Survived", FEATURES, titanic_df)

compare_results()


def linear_discriminant_fn(x_train, y_train, solver = "svd"):
    model = LinearDiscriminantAnalysis(solver= solver)
    model.fit(x_train, y_train)
    
    return model


result_dict["survived - linear_discriminant"] = build_model(linear_discriminant_fn, 
                                                            "Survived", 
                                                            # done to avoid dummy trap, droping one one hot encoded col
                                                            FEATURES[0:-1], 
                                                            titanic_df)

compare_results()


# quadratic discriminant analyis
# used only when covariance are diff for X for diff Y values
def quadratic_discriminant_fn(x_train, y_train):
    model = QuadraticDiscriminantAnalysis()
    model.fit(x_train, y_train)
    
    return model


result_dict["survived - quadratic_discriminant"] = build_model(quadratic_discriminant_fn, 
                                                            "Survived", 
                                                            # done to avoid dummy trap, droping one one hot encoded col
                                                            FEATURES[0:-1], 
                                                            titanic_df)

compare_results()


def sgd_fn(x_train, y_train, max_iter = 10000, tol=1e-3):
    model = SGDClassifier(max_iter=max_iter, tol = tol)
    model.fit(x_train, y_train)
    return model


result_dict["survived - SGD"] = build_model(sgd_fn, 
                                                            "Survived", 
                                                            # done to avoid dummy trap, droping one one hot encoded col
                                                            FEATURES[0:-1], 
                                                            titanic_df)

compare_results()


def linear_svc_fn(x_train, y_train, C=1.0, max_iter = 1000, tol=1e-3):
    model = LinearSVC(C = C, max_iter=max_iter, tol = tol, dual = False)
    model.fit(x_train, y_train)
    return model


result_dict["survived - linear_svc"] = build_model(linear_svc_fn, 
                                                            "Survived", 
                                                            # done to avoid dummy trap, droping one one hot encoded col
                                                            FEATURES[0:-1], 
                                                            titanic_df)

compare_results()


# Raidius neighbours clf
def radius_neighbor_fn(x_train, y_train, radius = 30.0):
    model = RadiusNeighborsClassifier(radius = radius)
    model.fit(x_train, y_train)
    return model


result_dict["survived - radius_neighbors"] = build_model(radius_neighbor_fn, 
                                                            "Survived", 
                                                            # done to avoid dummy trap, droping one one hot encoded col
                                                            FEATURES[0:-1], 
                                                            titanic_df)

compare_results()


def decision_tree_fn(x_train, y_train, max_depth=3, max_features=None):
    model = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features)
    model.fit(x_train, y_train)
    return model


result_dict["survived - decision tree"] = build_model(decision_tree_fn, 
                                                            "Survived", 
                                                            # done to avoid dummy trap, droping one one hot encoded col
                                                            FEATURES[0:-1], 
                                                            titanic_df)

compare_results()


# naive bayes
def naive_bayes_fn(x_train, y_train, priors = None):
    model = GaussianNB(priors=priors)
    model.fit(x_train, y_train)
    return model


result_dict["survived - naive bayes"] = build_model(naive_bayes_fn, 
                                                            "Survived", 
                                                            # done to avoid dummy trap, droping one one hot encoded col
                                                            FEATURES[0:-1], 
                                                            titanic_df)

compare_results()















