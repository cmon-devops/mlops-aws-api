# train.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
iris = load_iris()

# Train model
clf = RandomForestClassifier()
clf.fit(iris.data, iris.target)

# Save model
joblib.dump(clf, 'model.pkl')
