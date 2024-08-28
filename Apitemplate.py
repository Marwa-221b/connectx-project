#Import the libraries that you need Joblib is for the model , json and flask are for the api and the other libraries here are for preprocessing or any use

import joblib
import json
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#add the model directory after saving it
loaded_model = joblib.load('D:\model\car_model.ipynb') #model name

# Create a Flask app
app = Flask(__name__)

# Define a route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form as a json file so put all the fields that the model needs in the form

        data = request.get_json() #data get from website
        # Encode the categorical data
         #This step is for encoding if you have any categorical data that you encoded use it otherwise dont
        encoded_data = le.fit_transform(data)

        # Preprocess the data (You probably won't use it)
        for column, mapping in encoded_data.items():
            if column in data:
                data[column] = mapping[data[column]]

        #The data that came from the form in the format of an array to feed to the model
        data_array = [data['Make'], data['Model'], data['Engine'], data['Year'], data['Kilometer'],
                      data['Transmission'], data['Location'], data['Color'], data['Owner'],
                      data['Max Power'], data['Max Torque'], data['Drivetrain'],data['Lenght'],data['Width'],
                       data['Height'],data['Seating Capacity'],data['Fuel Tank Capacity']]
        data_array = [float(x) for x in data_array]  # Ensure data types are appropriate for your model
        
        #Reshaping the data (Don't change that line)
        data_array = np.array(data_array).reshape(1, -1) #check numercial data

        #Using the model you will use it like any other model to predict the input that came from the form.
        probabilities = loaded_model.predict_proba(data_array)

        percentage = probabilities[0][1] * 100

        #The return value of the api to show on the website
        return jsonify({'percentage': percentage})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)