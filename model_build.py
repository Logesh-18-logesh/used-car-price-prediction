import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

data=pd.read_excel(r"D:\websort\used car prediction\cardataset.xlsx")

# Ensure Mileage is treated as string
data['Mileage'] = data['Mileage'].astype(str)  # Convert to string to handle all cases

# Convert Mileage to float
data['Mileage'] = data['Mileage'].str.replace(' km/kg', '').str.replace(' kmpl', '')
data['Mileage'] = pd.to_numeric(data['Mileage'], errors='coerce')  # Converts to NaN for invalid entries

# Convert Engine to int
data['Engine'] = data['Engine'].replace('null', np.nan)  # Replace 'null' with NaN
data['Engine'] = data['Engine'].str.replace(' CC', '').astype(float)

# Replace 'null' and other invalid entries with NaN, then convert Power to float
data['Power'] = data['Power'].replace('null', np.nan)  # Replace 'null' with NaN
data['Power'] = data['Power'].astype(str).str.replace(' bhp', '')  # Ensure it is string before replacing
data['Power'] = pd.to_numeric(data['Power'], errors='coerce')  # Convert to float


data=data.dropna()

label_encoders = {}
for column in ['Name', 'Location', 'Fuel_Type', 'Transmission', 'Owner_Type']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le


X = data.drop('Price', axis=1)
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


joblib.dump(model, 'car_price_model.pkl')  
joblib.dump(label_encoders, 'label_encoders.pkl')  

