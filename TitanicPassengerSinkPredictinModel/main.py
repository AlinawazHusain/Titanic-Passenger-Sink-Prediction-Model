import numpy as np
import pandas as pd

        # Creating testing and training datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
x_test = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].values
y_train = train_data['Survived'].values
x_train = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].values

# y_test
test_dataY = pd.read_csv('gender_submission.csv')
y_test = test_dataY['Survived'].values

# data processing

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trans = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, ])], remainder='passthrough')
x_train = np.array(trans.fit_transform(x_train))
x_test = np.array(trans.fit_transform(x_test))

# filling NAN

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x_train[:, :])
x_train[:, :] = imputer.transform(x_train[:, :])
imputer.fit(x_test[:, :])
x_test[:, :] = imputer.transform(x_test[:, :])

# model
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

prediction = knn.predict(x_test)

from sklearn.metrics import accuracy_score

score = accuracy_score(prediction, y_test)
print(score)