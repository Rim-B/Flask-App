import pandas as pd
import numpy as np
import pickle

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target


X = boston.drop(['MEDV'], axis=1).values
Y = boston['MEDV'].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)


reg = LinearRegression().fit(X_train, Y_train)

with open('model.pkl','wb') as f:
    pickle.dump(reg,f)
    