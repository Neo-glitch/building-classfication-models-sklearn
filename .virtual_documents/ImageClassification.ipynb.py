import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier


fashion_mnist_df = pd.read_csv("./datasets/fashion-mnist_train.csv")
fashion_mnist_df.head()


# takes 40% of fashion mnist data to train our model
fashion_mnist_df = fashion_mnist_df.sample(frac=0.3).reset_index(drop=True)


# dict to map label num to actual category
LOOKUP = {
    0: "T-shirt",
    1: "Trouser",
    2: "PullOver",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}


# helper function to display an image and also the label in cat form
def display_image(features, actual_label):
    print(f"Actual label: {LOOKUP[actual_label]}")
    plt.imshow(features.reshape(28,28))


X = fashion_mnist_df[fashion_mnist_df.columns[1:]]
Y = fashion_mnist_df.label


# use defined helper fn to display image
display_image(X.loc[15].values, Y.loc[15])


# normalize immage
X /= 255.
X.head()


# log reg works here since image is grayScale bt dealing with RGB images best to work with NN(CNN)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state= 0)


# helper function that prints metrics score based on prediction and actual target passed
def summarize_classification(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred, normalize = True)   # since norm = True, acc in term of fraction
    num_acc = accuracy_score(y_test, y_pred, normalize = False)
    
    prec = precision_score(y_test, y_pred, average="weighted")  # since it's multiclass clf
    recall = recall_score(y_test, y_pred, average="weighted")
    
    print(f"accuracy count: {num_acc}")
    print(f"accuracy_score: {acc}")
    print(f"precision_score: {prec}")
    print(f"recall score: {recall}")
    print()


# solver is 'sag' is it converges faster when delaing with large datasets
logistic_model = LogisticRegression(solver="sag", multi_class="auto", max_iter=6000).fit(x_train, y_train)


















































































































