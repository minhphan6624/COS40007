import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

concrete = pd.read_csv("selected_converted_concrete.csv")

feature_cols = concrete.columns[:-1] # Features

print(feature_cols)

x = concrete[feature_cols] # Features
y = concrete["converted_strength"]    # Target variable

# Split dataset into training set and test set (70/30)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=1) 

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

#  Train the model using the training sets
clf = clf.fit(X_train,Y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

