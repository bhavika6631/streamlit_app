import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import os
from PIL import Image
import json

# --- This MUST be the first Streamlit command! ---
st.set_page_config(page_title="Land & Crop Recommendation", layout="centered")
# --- END of first command ---

# --- Global Configurations ---
IMG_HEIGHT, IMG_WIDTH = 64, 64
AGRICULTURAL_CLASSES = ['AnnualCrop', 'PermanentCrop', 'Pasture']

EURO_SAT_CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

# --- Model Loading Section (Cached for Efficiency) ---
@st.cache_resource
def load_all_models():
    """Loads all CNN models, RF model, scaler, and label encoder from the GitHub repo."""
    st.write("Loading models...") # Use st.write for user feedback

    loaded_models = {}

    # Load CNN models (assuming they are in the root of the GitHub repo)
    try:
        # File paths are relative to where app.py is located in the GitHub repo
        loaded_models['vgg16'] = tf.keras.models.load_model('VGG16_best_model.keras')
        st.success("VGG16 model loaded.")
    except Exception as e:
        st.error(f"Error loading VGG16 model: {e}. Ensure 'VGG16_best_model.keras' is in your GitHub repo's root.")
        loaded_models['vgg16'] = None

    try:
        loaded_models['vgg19'] = tf.keras.models.load_model('VGG19_best_model.keras')
        st.success("VGG19 model loaded.")
    except Exception as e:
        st.error(f"Error loading VGG19 model: {e}. Ensure 'VGG19_best_model.keras' is in your GitHub repo's root.")
        loaded_models['vgg19'] = None

    try:
        loaded_models['resnet50'] = tf.keras.models.load_model('ResNet50_best_model.keras')
        st.success("ResNet50 model loaded.")
    except Exception as e:
        st.error(f"Error loading ResNet50 model: {e}. Ensure 'ResNet50_best_model.keras' is in your GitHub repo's root.")
        loaded_models['resnet50'] = None

    # Load Crop Recommendation components (assuming they are in the root of the GitHub repo)
    try:
        loaded_models['rf_model'] = joblib.load('best_rf_model.joblib')
        loaded_models['scaler'] = joblib.load('crop_scaler.joblib')
        loaded_models['label_encoder'] = joblib.load('crop_label_encoder.joblib')
        st.success("Crop Recommendation components loaded.")
    except Exception as e:
        st.error(f"Error loading Crop Recommendation components: {e}. Ensure all .joblib files are in your GitHub repo's root.")
        loaded_models['rf_model'], loaded_models['scaler'], loaded_models['label_encoder'] = None, None, None

    return loaded_models

# Call the model loading function
models = load_all_models()
trained_vgg16 = models.get('vgg16')
trained_vgg19 = models.get('vgg19')
trained_resnet50 = models.get('resnet50')
best_rf_model = models.get('rf_model')
scaler = models.get('scaler')
crop_label_encoder = models.get('label_encoder')

# Import preprocess functions AFTER tensorflow is loaded to avoid errors if TF fails
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input

def predict_land_cover(image_data, model, preprocess_func, class_names, img_height, img_width):
    if model is None:
        return "Model Not Loaded", 0.0

    try:
        img = Image.open(image_data).convert('RGB').resize((img_width, img_height))
        img_array = np.array(img)
        img_preprocessed = preprocess_func(np.expand_dims(img_array, axis=0))

        predictions = model.predict(img_preprocessed, verbose=0)
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = class_names[predicted_class_index]
        confidence = np.max(predictions) * 100

        return predicted_class_name, confidence
    except Exception as e:
        st.error(f"An error occurred during land cover prediction: {e}")
        return "Prediction Error", 0.0

def recommend_crop(predicted_class_name, rf_model, feature_scaler, crop_le, soil_climate_data):
    if not (rf_model and feature_scaler and crop_le):
        st.warning("Crop recommendation models not loaded. Cannot recommend crop.")
        return None

    if predicted_class_name in AGRICULTURAL_CLASSES:
        st.subheader("Crop Recommendation:")
        st.write(f"Detected as '{predicted_class_name}' (agricultural land).")

        soil_climate_features = np.array([[
            soil_climate_data['N'],
            soil_climate_data['P'],
            soil_climate_data['K'],
            soil_climate_data['temperature'],
            soil_climate_data['humidity'],
            soil_climate_data['ph'],
            soil_climate_data['rainfall']
        ]])

        try:
            # Ensure the scaler is compatible with the input shape or data type
            soil_climate_features_scaled = feature_scaler.transform(soil_climate_features)
        except Exception as e:
            st.error(f"Error during scaling: {e}. Check if scaler was fitted correctly and input shape matches.")
            # Fallback to unscaled features if scaling fails, but warn the user
            soil_climate_features_scaled = soil_climate_features


        try:
            rf_prediction_encoded = rf_model.predict(soil_climate_features_scaled)
            recommended_crop = crop_le.inverse_transform(rf_prediction_encoded)[0]

            st.success(f"Based on soil/climate conditions, the recommended crop is: **{recommended_crop.upper()}**")
            return recommended_crop
        except Exception as e:
            st.error(f"Error during crop recommendation prediction: {e}")
            st.warning("Ensure input features for RF model are correct and compatible.")
            return None
    else:
        st.info(f"'{predicted_class_name}' is not an agricultural land type. No crop recommendation needed.")
        return None

st.title(" 🛰️ Land Cover Classification & Crop Recommendation")
st.markdown("Upload a satellite image to classify land cover, and provide soil data for crop recommendations.")

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tif"])

st.sidebar.header("Soil & Climate Data")
col1, col2 = st.sidebar.columns(2)
with col1:
    n_val = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=90, step=1)
    p_val = st.number_input("Phosphorus (P)", min_value=0, max_value=140, value=42, step=1)
    k_val = st.number_input("Potassium (K)", min_value=0, max_value=205, value=43, step=1)
    temp_val = st.number_input("Temperature (°C)", min_value=0.0, max_value=45.0, value=20.88, step=0.01)
with col2:
    humidity_val = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.00, step=0.01)
    ph_val = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=6.50, step=0.01)
    rainfall_val = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=202.94, step=0.01)

soil_climate_data_input = {
    'N': n_val, 'P': p_val, 'K': k_val, 'temperature': temp_val,
    'humidity': humidity_val, 'ph': ph_val, 'rainfall': rainfall_val
}

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")

    if st.button("Predict Land Cover & Recommend Crop"):
        st.subheader("Land Cover Classification Results:")
        with st.spinner('Classifying land cover and recommending crops...'):
            predicted_label_vgg16, confidence_vgg16 = predict_land_cover(
                uploaded_file, trained_vgg16, vgg16_preprocess_input, EURO_SAT_CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH
            )
            st.write(f"**VGG16 Prediction:** {predicted_label_vgg16} ({confidence_vgg16:.2f}% confidence)")

            predicted_label_vgg19, confidence_vgg19 = predict_land_cover(
                uploaded_file, trained_vgg19, vgg19_preprocess_input, EURO_SAT_CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH
            )
            st.write(f"**VGG19 Prediction:** {predicted_label_vgg19} ({confidence_vgg19:.2f}% confidence)")

            predicted_label_resnet50, confidence_resnet50 = predict_land_cover(
                uploaded_file, trained_resnet50, resnet50_preprocess_input, EURO_SAT_CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH
            )
            st.write(f"**ResNet50 Prediction:** {predicted_label_resnet50} ({confidence_resnet50:.2f}% confidence)")

            st.markdown("---")
            recommend_crop(predicted_label_resnet50, best_rf_model, scaler, crop_label_encoder, soil_climate_data_input)
else:
    st.info("Please upload an image to start the classification and recommendation process.")

st.markdown("---")
st.markdown("Developed with Streamlit and your pre-trained models.")
