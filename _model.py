import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('RTA_Dataset.csv')

df.drop(['Time','Day_of_week'],axis=1,inplace=True)

col =['Age_band_of_driver', 'Sex_of_driver',
       'Educational_level', 'Vehicle_driver_relation', 'Driving_experience',
       'Type_of_vehicle', 'Owner_of_vehicle', 'Service_year_of_vehicle',
       'Defect_of_vehicle', 'Area_accident_occured', 'Lanes_or_Medians',
       'Road_allignment', 'Types_of_Junction', 'Road_surface_type',
       'Road_surface_conditions', 'Light_conditions', 'Weather_conditions',
       'Type_of_collision', 'Number_of_vehicles_involved',
       'Number_of_casualties', 'Vehicle_movement', 'Casualty_class',
       'Sex_of_casualty', 'Age_band_of_casualty', 'Casualty_severity',
       'Work_of_casuality', 'Fitness_of_casuality', 'Pedestrian_movement',
       'Cause_of_accident', 'Accident_severity']

df = df.dropna(subset=['Defect_of_vehicle'])
df = df.dropna(subset=['Service_year_of_vehicle'])
df = df.dropna(subset=['Work_of_casuality'])

df['Educational_level'] = df['Educational_level'].fillna(df['Educational_level'].mode().iloc[0])
df['Vehicle_driver_relation'] = df['Vehicle_driver_relation'].fillna(df['Vehicle_driver_relation'].mode().iloc[0])
df['Driving_experience'] = df['Driving_experience'].fillna(df['Driving_experience'].mode().iloc[0])
df['Type_of_vehicle'] = df['Type_of_vehicle'].fillna(df['Type_of_vehicle'].mode().iloc[0])
df['Owner_of_vehicle'] = df['Owner_of_vehicle'].fillna(df['Owner_of_vehicle'].mode().iloc[0])
df['Area_accident_occured'] = df['Area_accident_occured'].fillna(df['Area_accident_occured'].mode().iloc[0])
df['Lanes_or_Medians'] = df['Lanes_or_Medians'].fillna(df['Lanes_or_Medians'].mode().iloc[0])
df['Road_allignment'] = df['Road_allignment'].fillna(df['Road_allignment'].mode().iloc[0])
df['Types_of_Junction'] = df['Types_of_Junction'].fillna(df['Types_of_Junction'].mode().iloc[0])
df['Road_surface_type'] = df['Road_surface_type'].fillna(df['Road_surface_type'].mode().iloc[0])
df['Type_of_collision'] = df['Type_of_collision'].fillna(df['Type_of_collision'].mode().iloc[0])
df['Vehicle_movement'] = df['Vehicle_movement'].fillna(df['Vehicle_movement'].mode().iloc[0])
df['Fitness_of_casuality'] = df['Fitness_of_casuality'].fillna(df['Fitness_of_casuality'].mode().iloc[0])


columns = ['Age_band_of_driver', 'Sex_of_driver', 'Educational_level',
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
for i in columns:
    df[i] = le.fit_transform(df[i])

with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(le, le_file)

colum = ['Educational_level','Vehicle_driver_relation', 'Driving_experience', 'Type_of_vehicle',
       'Owner_of_vehicle', 'Service_year_of_vehicle', 'Defect_of_vehicle',
       'Area_accident_occured', 'Lanes_or_Medians', 'Road_allignment',
       'Types_of_Junction', 'Road_surface_type','Number_of_casualties',
       'Vehicle_movement','Work_of_casuality','Fitness_of_casuality', 'Pedestrian_movement', 'Cause_of_accident']

for i in colum:
    df.drop(i,axis=1,inplace=True)

X = df.drop('Accident_severity', axis=1)
y = df['Accident_severity']

sm=SMOTE(random_state=10)
x_new,y_new=sm.fit_resample(X,y)

X_train,X_test,y_train,y_test=train_test_split(x_new,y_new,test_size=0.3,random_state=1)

rf = RandomForestClassifier()

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(rf, model_file)

