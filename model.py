import numpy as np
import pandas as pd
import pickle
from sklearn import tree

# Loading dataset
FileCSV="diabetes.csv"
df_diabetes = pd.read_csv(FileCSV,sep=",")

X = df_diabetes.iloc[:, 0:-1].values
y = df_diabetes.iloc[:, -1].values

# Initialization and fitting of the model
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf.fit(X, y)

# Serialization
pickle.dump(clf, open('model.pkl','wb'))

# Deserialization
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[6, 148, 72, 35, 0, 33.6, 0.627, 50]]))
