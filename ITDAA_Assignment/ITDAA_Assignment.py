#ITDAA4-12 QUESTION 1
#Importing the libraries needed 
import pandas as pd   #data manipulation
import matplotlib.pyplot as plt   #visualizing the data
import sqlite3   #interacting with the SQL database
import seaborn as sns   #complex graphs and visualization

#Reading the csv file given and storing it in the pandas database
heart_df = pd.read_csv('heart.csv')

#Split columns will seperate the single column from csv file into different columns where there is a semicolon
split_columns = heart_df['age;sex;cp;trestbps;chol;fbs;restecg;thalach;exang;oldpeak;slope;ca;thal;target'].str.split(';', expand=True)
heart_df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']] = split_columns

#Removing the column with all the fields together and it won't be represented in the dataframe
heart_df.drop(columns=['age;sex;cp;trestbps;chol;fbs;restecg;thalach;exang;oldpeak;slope;ca;thal;target'], inplace=True)

#Connecting to the database created caled heart.db
with sqlite3.connect('heart.db') as conn:
    #Writing data to database and checking table exists
    heart_df.to_sql('heart_table', conn, if_exists='replace', index=False)

#QUESTION 2.1.a
#Connecting to the database created in QUESTION 1
with sqlite3.connect('heart.db') as conn:
    #Selecting all the data from the table and reading it
    query = "SELECT * FROM heart_table"
    heart_df = pd.read_sql(query,conn)

#Calculating the amount of missing values that each column has and prints it out
values_missing = heart_df.isnull().sum()
print("Missing Values:\n", values_missing)

#Selecting and converting columns to numeric data types
num_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'] 
heart_df[num_columns] = heart_df[num_columns].apply(pd.to_numeric, errors='coerce')
heart_df.fillna(heart_df.mean(), inplace=True)

#QUESTION 2.1.b
#Defining list using the names of columns
categ_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
#Creating a grid 
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
#Flattening the grid to a 1 Dimension array
axes = axes.flatten()
#Getting index for each column
for i, column in enumerate(categ_columns):
    #Creating the plot for the graphs
    sns.countplot(x=column, hue='target', data=heart_df, ax=axes[i])
    #Setting the tittle for each of the graphs
    axes[i].set_title(f'Distribution of {column} by Target')
    #Setting the X label for each of the graphs
    axes[i].set_xlabel(column)
    #Setting the Y label for each of the graphs
    axes[i].set_ylabel('Count')
#Adjusting the layout of the subplots
plt.tight_layout()
plt.show()

#QUESTION 2.1.c
#Defining the list of names
num_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
#Creating and setting up figure
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
#Flattening the grid to a 1 Dimension array
axes = axes.flatten()
#Getting index for each column
for i, column in enumerate(num_columns):
    #Creating the plot for the graphs
    sns.boxplot(x='target', y=column, data=heart_df, ax=axes[i])
    #Setting the tittle for each of the graphs
    axes[i].set_title(f'Distribution of {column} by Target')
    #Setting the X label for each of the graphs
    axes[i].set_xlabel('Target')
    #Setting the Y label for each of the graphs
    axes[i].set_ylabel(column)
#Adjusting the layout of the subplots    
plt.tight_layout()
plt.show()

#QUESTION 3
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#QUESTION 3.1
features = heart_df.drop(columns=['target'])
target = heart_df['target']

#Names of categorical features in the dataset in a list
categ_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
#Names of numerical features in the dataset in a list
num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

#Inserting missing values by changing the format and using the value that appears the most in respective columns
categ_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder())
])

#Inserting missing values by standardizing the numerical values that appear the most in respective columns
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

#Preprocessing procedures being integrated for categorical and numerical features 
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categ_transformer, categ_features),
        ('num', num_transformer, num_features)
    ])

#Splitting data into a training set and testing set 
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

#QUESTION 3.2
#Creating an instance of a RandomForestClassifier from the scikit-learn library
random_forest = RandomForestClassifier(random_state=42)
#Creating an instance of the Support Vector Machine (SVM)
svm = SVC(kernel='rbf', random_state=42)
#Creating an instance of the Gradient Boosting Classifier
gradient_boosting_classifier = GradientBoostingClassifier(random_state=42)


random_forest.fit(X_train_processed, y_train)  #Training the Random Forest classifier
svm.fit(X_train_processed, y_train)  #Training the Support Vector Machine classifier
gradient_boosting_classifier.fit(X_train_processed, y_train)  #Training the Gradient Boosting classifier

rf_predictions = random_forest.predict(X_test_processed)  #Using the trained Random Forest classifier for predictions
svm_predictions = svm.predict(X_test_processed)  #Using the trained Support Vector Machine classifier for predictions
gb_predictions = gradient_boosting_classifier.predict(X_test_processed)  #Using the trained Gradient Boosting classifier for predictions

rf_accuracy = accuracy_score(y_test, rf_predictions)  #Calculating the accuracy of the Random Forest classifier's predictions
svm_accuracy = accuracy_score(y_test, svm_predictions)  #Calculating the accuracy of the Support Vector Machine classifier's predictions
gb_accuracy = accuracy_score(y_test, gb_predictions)  #Calculating the accuracy of the Gradient Boosting classifier's predictions

#Printing the results
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"SVM Accuracy: {svm_accuracy:.4f}")
print(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}")

#Checking if the Gradient Boosting is most accurate 
if gb_accuracy > rf_accuracy and gb_accuracy > svm_accuracy:
    best_model = gradient_boosting_classifier
    #If true, the model is saved as a file 
    best_model_name = "Gradient_Boosting_Model.pkl"
    joblib.dump(best_model, best_model_name)
    #Displaying that the best model has been saved to disk
    print(f"Saved the best model ({best_model_name}) to disk.")
#Checking if the Random Forrest is most accurate
elif rf_accuracy > svm_accuracy:
    best_model = random_forest
    #If true, the model is saved as a file 
    best_model_name = "Random_Forest_Model.pkl"
    joblib.dump(best_model, best_model_name)
    #Displaying that the best model has been saved to disk
    print(f"Saved the best model ({best_model_name}) to disk.")
#Checking if the Support Vector Machine is most accurate
else:
    best_model = svm
    #If true, then the model is saved as a file 
    best_model_name = "SVM_Model.pkl"
    joblib.dump(best_model, best_model_name)
    #Displaying that the best model has been saved to disk
    print(f"Saved the best model ({best_model_name}) to disk.")


#QUESTION 4
#Importing necessary libraries
import streamlit as st  #Streamlit for creating the web app
import pandas as pd  #pandas for data manipulation
import joblib  #joblib for loading the trained model
from sklearn.preprocessing import StandardScaler  #StandardScaler for feature scaling

#Loading the trained SVM model from a file
model = joblib.load('SVM_Model.pkl')
#Creating an instance of StandardScaler for scaling the input data
data_scaler = StandardScaler()

#Setting the title of the Streamlit app
st.title('Heart Disease Prediction')
#Writing a description of the app
st.write('Enter patient details to predict if they have heart disease.')

#Function to preprocess the input data
def preprocess_input_data(patient_age, patient_sex, chest_pain, resting_bp, cholesterol, fasting_bs, ecg_result, max_heart_rate, exercise_angina, st_depression, st_slope, major_vessels, thalassemia):
    patient_sex = 1 if patient_sex == 'Male' else 0  #Converting sex to binary value (1 for Male, 0 for Female)
    fasting_bs = 1 if fasting_bs == 'True' else 0  #Converting fasting blood sugar to binary value (1 for True, 0 for False)

    #Creating a DataFrame with the input data
    input_data = pd.DataFrame({
        'age': [patient_age],
        'sex': [patient_sex],
        'cp': [chest_pain],
        'trestbps': [resting_bp],
        'chol': [cholesterol],
        'fbs': [fasting_bs],
        'restecg': [ecg_result],
        'thalach': [max_heart_rate],
        'exang': [exercise_angina],
        'oldpeak': [st_depression],
        'slope': [st_slope],
        'ca': [major_vessels],
        'thal': [thalassemia]
    })
    return input_data

#Input fields for the user to enter patient details
patient_age = st.number_input('Age', min_value=1, max_value=120)  #Age input field
patient_sex = st.radio('Sex', ['Male', 'Female'])  #Sex input field
chest_pain = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])  #Chest pain type input field
resting_bp = st.number_input('Resting Blood Pressure (mmHg)', min_value=1)  #Resting blood pressure input field
cholesterol = st.number_input('Cholesterol (mg/dl)', min_value=1)  #Cholesterol input field
fasting_bs = st.radio('Fasting Blood Sugar > 120 mg/dl', ['True', 'False'])  #Fasting blood sugar input field
ecg_result = st.selectbox('Resting ECG Result', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])  #Resting ECG result input field
max_heart_rate = st.number_input('Max Heart Rate Achieved', min_value=1)  #Max heart rate achieved input field
exercise_angina = st.radio('Exercise Induced Angina', ['Yes', 'No'])  #Exercise induced angina input field
st_depression = st.number_input('ST Depression Induced by Exercise Relative to Rest', min_value=0.0)  #ST depression input field
st_slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])  #Slope of the peak exercise ST segment input field
major_vessels = st.selectbox('Number of Major Vessels Colored by Flourosopy', [0, 1, 2, 3])  #Number of major vessels input field
thalassemia = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])  #Thalassemia input field

#Predict button
if st.button('Predict'):
    #Preprocessing the input data
    input_data = preprocess_input_data(patient_age, patient_sex, chest_pain, resting_bp, cholesterol, fasting_bs, ecg_result, max_heart_rate, exercise_angina, st_depression, st_slope, major_vessels, thalassemia)
    #Scaling the input data
    input_data_scaled = data_scaler.fit_transform(input_data)
    #Making predictions using the loaded model
    prediction = model.predict(input_data_scaled) 

    #Displaying the prediction result
    if prediction[0] == 1:
        st.success('The patient is likely to have heart disease.')
    else:
        st.success('The patient is unlikely to have heart disease.')
