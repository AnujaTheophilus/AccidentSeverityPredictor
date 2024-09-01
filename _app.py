from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

feature_columns = [
    'Age_band_of_driver', 'Sex_of_driver', 'Road_surface_conditions', 'Light_conditions',
    'Weather_conditions', 'Type_of_collision', 'Number_of_vehicles_involved', 'Casualty_class',
    'Sex_of_casualty', 'Age_band_of_casualty', 'Casualty_severity'
]

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = [request.form.get(col, '') for col in feature_columns]
    number_of_vehicles_involved = int(data[feature_columns.index('Number_of_vehicles_involved')])
    data[feature_columns.index('Number_of_vehicles_involved')] = number_of_vehicles_involved
    le = pickle.load(open('label_encoder.pkl', 'rb'))
    transformed_data = []
    for i, col in enumerate(feature_columns):
        if col in ['Age_band_of_driver', 'Sex_of_driver', 'Road_surface_conditions', 'Light_conditions',
                   'Weather_conditions', 'Type_of_collision', 'Casualty_class', 'Sex_of_casualty',
                   'Age_band_of_casualty', 'Casualty_severity']:
            transformed_data.append(le.fit_transform([data[i]])[0])
        else:
            transformed_data.append(data[i])

    input_array = [transformed_data]

    new_pred = model.predict(input_array)

    prediction_label = label_encoder.inverse_transform(new_pred)[0]

    return render_template('result.html', prediction=prediction_label)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
