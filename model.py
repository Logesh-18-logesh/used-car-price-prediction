import pandas as pd
import numpy as np
import joblib

def load_model():
    """Load the trained model"""
    model = joblib.load('car_price_model.pkl')  # Save this model after training
    return model

def load_label_encoders():
    """Load label encoders used during model training"""
    return joblib.load('label_encoders.pkl')  # Save the encoders after training

def predict_car_price(car_data, model, label_encoders):
    """Transform user inputs and make predictions"""
    
    # Encode categorical fields using saved label encoders
    car_data['Name'] = label_encoders['Name'].transform([car_data['Name']])[0]
    car_data['Location'] = label_encoders['Location'].transform([car_data['Location']])[0]
    car_data['Fuel_Type'] = label_encoders['Fuel_Type'].transform([car_data['Fuel_Type']])[0]
    car_data['Transmission'] = label_encoders['Transmission'].transform([car_data['Transmission']])[0]
    car_data['Owner_Type'] = label_encoders['Owner_Type'].transform([car_data['Owner_Type']])[0]

    # Convert dictionary to DataFrame for prediction
    input_df = pd.DataFrame([car_data])

    # Predict the price
    predicted_price = model.predict(input_df)[0]
    return round(predicted_price, 2)
