import pandas as pd

titanic_df = pd.read_csv("./datasets/titanic_train.csv")

titanic_df.head()


# drop irrelevant cols
titanic_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], "columns", inplace=True)
titanic_df.head()


# check each col in df for nan values
titanic_df[titanic_df.isnull().any(axis = 1)].count()


titanic_df.dropna(inplace=True)


titanic_df.describe()


# viz how feature affect target(Survived)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize = (12, 8))
plt.scatter(titanic_df.Fare, titanic_df.Survived)
plt.xlabel("Age")
plt.ylabel("Survived")


pd.crosstab(titanic_df.Sex, titanic_df.Survived)


pd.crosstab(titanic_df.Pclass, titanic_df.Survived)


titanic_data_corr = titanic_df.corr()

titanic_data_corr


import seaborn as sns


fig, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(titanic_data_corr, annot=True)


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
titanic_df.Sex = le.fit_transform(titanic_df.Sex.astype(str))  # 0 female and 1 is male

titanic_df.head()


# one hot encode Embarked
titanic_df = pd.get_dummies(titanic_df, columns= ["Embarked"])
titanic_df.head()


# shuffles the preprocesed dataset
titanic_df = titanic_df.sample(frac=1).reset_index(drop=True) # drop index

titanic_df.head()


titanic_df.to_csv("./datasets/titanic_mine.csv", index = False)  # saves dataset to csv


titanic_df = pd.read_csv("./datasets/titanic_mine.csv")
titanic_df.head()


from sklearn.model_selection import train_test_split

X = titanic_df.drop("Survived", axis = 1)
Y = titanic_df.Survived

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LogisticRegression

# C specifies inverse strenth of regularization( # lower is higher regul)
logisitc_model = LogisticRegression(penalty="l2", C= 1.0, solver = "liblinear").fit(X_train, y_train)


y_pred = logisitc_model.predict(X_test)


pred_results = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})

pred_results.head()


# conf matrix using pd instead of conf matrix lib
titanic_crosstab = pd.crosstab(pred_results.y_pred, pred_results.y_test)
titanic_crosstab


from sklearn.metrics import accuracy_score, precision_score, recall_score


acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {acc}")
print(f"\nPrecision: {prec}")
print(f"\nrecall: {recall}")


















