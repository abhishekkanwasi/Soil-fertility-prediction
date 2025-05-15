import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('soil_data.csv')

# Encode labels
le = LabelEncoder()
df['Fertility'] = le.fit_transform(df['Fertility'])
joblib.dump(le, 'label_encoder.pkl')

X = df.drop('Fertility', axis=1)
y = df['Fertility']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, 'soil_fertility_model.pkl')
print("Model and encoder saved.")
