# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load Dataset
file_path = 'WineQT.csv'  # Replace with your dataset path if different
data = pd.read_csv(file_path)

# Data Exploration
print("Dataset Info:\n", data.info())
print("\nFirst 5 Rows of the Dataset:\n", data.head())
print("\nSummary Statistics:\n", data.describe())

# Data Visualization
sns.histplot(data['quality'], kde=False, bins=6)
plt.title('Wine Quality Distribution')
plt.xlabel('Quality')
plt.ylabel('Frequency')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title('Correlation Heatmap')
plt.show()

# Data Preprocessing
X = data.drop('quality', axis=1)
y = data['quality']

# Convert quality to binary classification (Good: >=7, Bad: <7)
y = y.apply(lambda q: 1 if q >= 7 else 0)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training and Evaluation
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Stochastic Gradient Descent": SGDClassifier(random_state=42),
    "Support Vector Classifier": SVC(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Results:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

