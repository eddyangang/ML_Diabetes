# Diabetes Prediction using Machine Learning

## Description 
This simple web application predicts whether a patient has diabetes based on a variety of physiological attributes (i.e glucaose, blood pressure, insulin levels) by creating a training model using machine learning. The training data can be found [here](https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/diabetes.csv).

## Technologies used
- Python
- Streamlit
- Pandas
- Scikit

## Training Model
To create our model, I split the diabetes linked above, into 75% training, and 25% testing. 

I used a random forest classifier to generate a decision tree based on the sample dataset that used averageing to improve the predictive accuract and control over-fitting data. 
