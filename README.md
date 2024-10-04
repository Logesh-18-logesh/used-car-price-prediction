# USED CAR PRICE PREDICTION 
## USING RANDOM FOREST REGRESSOR

This repository contains code for building a price prediction model using random forest regressor , cod efor using the model and a simple frontend 

## What is RandomForestRegressor

A random forest is a meta estimator that fits a number of decision tree regressors on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. Trees in the forest use the best split strategy.

## HOW TO RUN THE FILE

1. Install the required libraries:

   >pandas
   >numpy
   >scikit-learn
   >flask
   >joblib

2. Make sure the files app.py,model_build.py,model.py,cardataset.xlsx and templates folder sre in same directory

3. At first run the model_build.py file. This creates two more .pkl files in the directory which are the models and the label encoders

4. Once you see the new files in the directory run the app.py file


## HOW THE CODE WORKS

### model_build.py:

This code is used to build and train a machine learning model to predict the price of a used car based on several features. It uses the Random Forest Regressor algorithm from scikit-learn and performs data preprocessing, model training, evaluation, and saving the trained model and label encoders.

### model.py:

This code provides the necessary functions to load a pre-trained machine learning model (a Random Forest Regressor) and its associated label encoders, and then use these to make car price predictions based on new input data provided by a user. 

### app.py:

This Flask application serves as a web-based interface for predicting car prices using a pre-trained machine learning model. 
