import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = {
    'study_hours': [5, 8, 3, 2, 7, 6, 9, 4, 10, 1],
    'attendance': [90, 80, 70, 60, 85, 95, 75, 65, 88, 55],
    'marks': [78, 85, 50, 40, 90, 80, 95, 60, 99, 35],
    'result': ['Pass', 'Pass', 'Fail', 'Fail', 'Pass', 'Pass', 'Pass', 'Fail', 'Pass', 'Fail']
}

df = pd.DataFrame(data)

df['result'] = df['result'].map({'Pass': 1, 'Fail': 0})

X = df[['study_hours', 'attendance', 'marks']]  
y = df['result']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Predictions: {y_pred}")
print(f"Actual: {y_test.values}")
print(f"Accuracy: {accuracy * 100:.2f}%")
