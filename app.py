from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('soil_fertility_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'Nitrogen': float(request.form['nitrogen']),
            'Phosphorus': float(request.form['phosphorus']),
            'Potassium': float(request.form['potassium']),
            'pH': float(request.form['ph']),
            'Temperature': float(request.form['temperature']),
            'Humidity': float(request.form['humidity']),
            'Rainfall': float(request.form['rainfall'])
        }
        input_df = pd.DataFrame([data])
        pred_encoded = model.predict(input_df)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        return render_template('index.html', prediction_text=f'Soil Fertility: {pred_label}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
