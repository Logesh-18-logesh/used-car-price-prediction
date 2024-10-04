from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
from model import predict_car_price, load_label_encoders, load_model

app = Flask(__name__)

# Load pre-trained model and label encoders
model = load_model()
label_encoders = load_label_encoders()

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    car_data = {
        'Name': request.form['name'],
        'Location': request.form['location'],
        'Year': int(request.form['year']),
        'Kilometers_Driven': int(request.form['kilometers_driven']),
        'Fuel_Type': request.form['fuel_type'],
        'Transmission': request.form['transmission'],
        'Owner_Type': request.form['owner_type'],
        'Mileage': float(request.form['mileage']),
        'Engine': float(request.form['engine']),
        'Power': float(request.form['power']),
        'Seats': int(request.form['seats'])
    }

    # Predict the car price
    predicted_price = predict_car_price(car_data, model, label_encoders)
    pc = str(predicted_price) + " lakhs"

    return jsonify({'predicted_price': pc})

if __name__ == '__main__':
    app.run(debug=True)
