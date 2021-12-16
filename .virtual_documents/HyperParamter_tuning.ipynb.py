import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


titanic_df = pd.read_csv("./datasets/titanic_mine.csv")
titanic_df.head()


X = titanic_df.drop("Survived", axis = 1)
Y = titanic_df.Survived

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)


# helper function that prints metrics score based on prediction and actual target passed
def summarize_classification(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred, normalize = True)   # since norm = True, acc in term of fraction
    num_acc = accuracy_score(y_test, y_pred, normalize = False)
    
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"accuracy count: {num_acc}")
    print(f"accuracy_score: {acc}")
    print(f"precision_score: {prec}")
    print(f"recall score: {recall}")
    print()


from sklearn.model_selection import GridSearchCV

param_grid = {"max_depth": [2, 4, 5, 7, 9, 10]}

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv = 3, return_train_score=True)
grid_search.fit(x_train, y_train)

grid_search.best_params_


# creates a de tree model using best max_depth ret by grid_search
decision_tree_model = DecisionTreeClassifier(
    max_depth = grid_search.best_params_["max_depth"]
).fit(x_train, y_train)


y_pred = decision_tree_model.predict(x_test)

# use helper function to get metrics for this model
summarize_classification(y_test, y_pred)





























































































