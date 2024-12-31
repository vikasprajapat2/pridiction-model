import streamlit as st
import numpy as np
import pickle

# Load the model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Streamlit UI
st.title("Iris Flower Classification")
st.write("Predict the type of Iris flower using sepal and petal dimensions.")

# Input Features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

# Prediction
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    species = ['Setosa', 'Versicolor', 'Virginica']
    st.write(f"Predicted Species: {species[prediction[0]]}")

