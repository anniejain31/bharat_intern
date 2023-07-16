# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Titanic dataset
data = pd.read_csv('bharat_intern/titanic_train.csv')

# Select relevant features and target variable
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

# Preprocess the data
data = data[features + [target]].dropna()  # Remove rows with missing values
data['Sex'] = LabelEncoder().fit_transform(data['Sex'])  # Convert 'Sex' to numerical

# Encode 'Embarked' column using one-hot encoding
embarked_encoded = pd.get_dummies(data['Embarked'], prefix='Embarked')
data = pd.concat([data, embarked_encoded], axis=1).drop('Embarked', axis=1)

# Split the data into training and testing sets
X = data.drop(target, axis=1)
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict the survival outcomes
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
