import pandas as pd
import numpy as np
import graphviz  # conda install python-graphviz
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

# Load the test_data set
td = pd.read_csv('Project7_2.csv')

# Convert the data to numpy arrays
X = np.delete(td.values, 3, axis=1)
y = np.delete(td.values, [0, 1, 2], axis=1)
# Fit doesn't like string data, so convert to floats
le = preprocessing.LabelEncoder()
for i in range(3):
    X[:, i] = le.fit_transform(X[:, i])
print(X)

# Run the classification tree
model = DecisionTreeClassifier()
model.fit(X, y)

dot_data = tree.export_graphviz(model, out_file=None)
graph = graphviz.Source(dot_data)
graph.render('test')
dot_data = tree.export_graphviz(model,
                                out_file=None,
                                feature_names=['Trend', 'Volume', 'Time'],
                                class_names=['Up', 'Down'],
                                filled=True,
                                rounded=True,
                                special_characters=True)

graph = graphviz.Source(dot_data)
predictions = model.predict(X)

print("\nAccuracy: ", accuracy_score(y, predictions) * 100, "%")
