# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv('soil_data.csv')  # Replace with your dataset path

# Display basic info
print("Dataset preview:")
print(df.head())

# Encode categorical target if necessary
le = LabelEncoder()
df['Fertility'] = le.fit_transform(df['Fertility'])

# Split features and target
X = df.drop('Fertility', axis=1)
y = df['Fertility']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, 'soil_fertility_model.pkl')
print("Model saved as soil_fertility_model.pkl")
