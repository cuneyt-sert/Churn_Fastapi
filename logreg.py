import pandas as pd

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



df = pd.read_csv('ChurnData.csv')


X_train, x_test, Y_train, y_test = train_test_split(df.iloc[:,1:7], df.iloc[:,0:1], 
                                                    test_size = 0.2, random_state=42)

std_scale = StandardScaler()

X_train_scaled = std_scale.fit_transform(X_train)

logreg1 = LogisticRegression(solver='liblinear',C=0.8)
logreg1_model=logreg1.fit(X_train_scaled, Y_train.values.ravel())

print(logreg1_model.predict([[25,15,10,5,1,1]]))

import pickle

filename = "logreg1_model.pkl"
pickle.dump(logreg1_model, open(filename, "wb"))

