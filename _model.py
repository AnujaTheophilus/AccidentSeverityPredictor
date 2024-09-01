import pandas as pd
import warnings
import pickle
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('RTA_Dataset.csv')
df.drop(['Time', 'Day_of_week'], axis=1, inplace=True)

# Drop rows with NaNs in specific columns
df = df.dropna(subset=['Defect_of_vehicle', 'Service_year_of_vehicle', 'Work_of_casuality'])

# Fill remaining NaNs with the mode of each column
fill_mode_columns = ['Educational_level', 'Vehicle_driver_relation', 'Driving_experience',
                     'Type_of_vehicle', 'Owner_of_vehicle', 'Area_accident_occured',
                     'Lanes_or_Medians', 'Road_allignment', 'Types_of_Junction',
                     'Road_surface_type', 'Type_of_collision', 'Vehicle_movement',
                     'Fitness_of_casuality']
for col in fill_mode_columns:
    df[col] = df[col].fillna(df[col].mode().iloc[0])

# Label encoding
columns_to_encode = ['Age_band_of_driver', 'Sex_of_driver', 'Educational_level',
                     'Vehicle_driver_relation', 'Driving_experience', 'Type_of_vehicle',
                     'Owner_of_vehicle', 'Service_year_of_vehicle', 'Defect_of_vehicle',
                     'Area_accident_occured', 'Lanes_or_Medians', 'Road_allignment',
                     'Types_of_Junction', 'Road_surface_type', 'Road_surface_conditions',
                     'Light_conditions', 'Weather_conditions', 'Type_of_collision',
                     'Vehicle_movement', 'Casualty_class', 'Sex_of_casualty',
                     'Age_band_of_casualty', 'Casualty_severity', 'Work_of_casuality',
                     'Fitness_of_casuality', 'Pedestrian_movement', 'Cause_of_accident',
                     'Accident_severity']

le = LabelEncoder()
for col in columns_to_encode:
    df[col] = le.fit_transform(df[col])

with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(le, le_file)

# Drop less important columns
columns_to_drop = ['Educational_level', 'Vehicle_driver_relation', 'Driving_experience',
                   'Type_of_vehicle', 'Owner_of_vehicle', 'Service_year_of_vehicle',
                   'Defect_of_vehicle', 'Area_accident_occured', 'Lanes_or_Medians',
                   'Road_allignment', 'Types_of_Junction', 'Road_surface_type',
                   'Number_of_casualties', 'Vehicle_movement', 'Work_of_casuality',
                   'Fitness_of_casuality', 'Pedestrian_movement', 'Cause_of_accident']
df.drop(columns=columns_to_drop, axis=1, inplace=True)

# Split data into features and target
X = df.drop('Accident_severity', axis=1)
y = df['Accident_severity']

# Handle class imbalance using SMOTE
sm = SMOTE(random_state=10)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=1)

# Train and evaluate models
rf = RandomForestClassifier()

with open('model.pkl', 'wb') as model_file:
    pickle.dump(rf, model_file)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("RandomForestClassifier Accuracy:", accuracy_score(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

