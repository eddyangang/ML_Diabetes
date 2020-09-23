# Description: This program detects if someone has diabetes using machine learning and python

# Import Libraries

import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st


# Create a title and subtitle
st.write("""
# Diabetes Detection
Detect if some has diabetes using machine learning and python !
""")

# Open and display an Image
image = Image.open('./machineLearning.png')
st.image(image, caption="Machine Learning With Python", use_column_width=True)

# Get the Data
df = pd.read_csv('C:/Users/Eddy/Documents/GitHub/ML_Diabetes/diabetes_data.csv', sep=",")

# Set a sub header
st.subheader("Data Information:")

# Show the data as a table
st.dataframe(df)

# Show statistics on the data
st.write(df.describe())

# Show the data as a chart
chart = st.bar_chart(df)

# Split the data into independent "X", and dependent "Y" variables
X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

# Split the data into 75% Training and 25% Testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Get the feature input from the user


def get_user_input():
    pregnancies = st.sidebar.slider("pregnancies", 0, 17, 3)
    glucose = st.sidebar.slider("glucose", 0, 199, 117)
    blood_pressure = st.sidebar.slider("blood_pressure", 0, 122, 72)
    skin_thickness = st.sidebar.slider("skin_thickness", 0, 99, 23)
    insulin = st.sidebar.slider("insulin", 0.0, 846.0, 30.0)
    bmi = st.sidebar.slider("bmi", 0.0, 67.1, 32.0)
    diabetes_pedigree_function = st.sidebar.slider("diabetes_pedigree_function", 0.078, 2.42, 0.3725)
    age = st.sidebar.slider("age", 21, 81, 29)

    # Store a dictionary into a variable
    user_data = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'bmi': bmi,
        'diabetes_pedigree_function': diabetes_pedigree_function,
        'age': age
    }

    # Transform the data into a data frame
    features = pd.DataFrame(user_data, index=[0])
    return features


# Store the user input into a variable
user_input = get_user_input()

# Set a sub header and display the users input
st.subheader('User Input:')
st.write(user_input)

# Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

# Show the model metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%')

# Store the models predictions in a variable
prediction = RandomForestClassifier.predict(user_input)
if prediction == 1:
    prediction = "According to the model, you likely have diabetes."
else:
    prediction = "According to the model, you likely do not have diabetes."
# Set a sub header and display the classification
st.subheader('Classification: ')
st.write(prediction)
