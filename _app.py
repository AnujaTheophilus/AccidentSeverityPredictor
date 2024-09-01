from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

# Load the trained model and RFE
model = pickle.load(open('model.pkl', 'rb'))
with open('label_encoder.pkl', 'rb') as le_file:
    le = pickle.load(le_file)

# Define the feature columns (these should match the ones used in training)
feature_columns = ['Time', 'Day_of_week', 'Age_band_of_driver', 'Sex_of_driver',
                   'Educational_level', 'Vehicle_driver_relation', 'Driving_experience',
                   'Type_of_vehicle', 'Owner_of_vehicle', 'Area_accident_occured',
                   'Lanes_or_Medians', 'Road_allignment', 'Types_of_Junction',
                   'Road_surface_type', 'Road_surface_conditions', 'Light_conditions',
                   'Weather_conditions', 'Type_of_collision', 'Vehicle_movement',
                   'Casualty_class', 'Pedestrian_movement', 'Cause_of_accident']


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = {col: request.form.get(col, '') for col in feature_columns}

    # Create a DataFrame from the form data
    new_data = pd.DataFrame([data], columns=feature_columns)

    le_new = LabelEncoder()

    for col in ['Age_band_of_driver', 'Sex_of_driver', 'Road_surface_conditions',
                'Light_conditions', 'Weather_conditions', 'Type_of_collision',
                'Casualty_class', 'Sex_of_casualty', 'Age_band_of_casualty', 'Casualty_severity']:
        new_data[col] = le_new.fit_transform(new_data[col])

    new_pred = model.predict(new_data)

    prediction_label = le.inverse_transform(new_pred)[0]

    return render_template('result.html', prediction=prediction_label)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
