import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import os
from PIL import Image
import gdown

# --- SET PAGE CONFIG ---
st.set_page_config(page_title="Land & Crop Recommendation", layout="centered")

# --- GLOBALS ---
IMG_HEIGHT, IMG_WIDTH = 64, 64
AGRICULTURAL_CLASSES = ['AnnualCrop', 'PermanentCrop', 'Pasture']
EURO_SAT_CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

# --- GOOGLE DRIVE FILE IDS ---
GDRIVE_FILES = {
    "vgg16":  "1unAd1fp1rjY6JLCyo9UOiLQVXdvTMJ2f",
    "vgg19":  "1HcCqPDnzsH0xKlAF8XM8wZ1rWcdw7nop",
    "resnet50": "1HLsAwy0944QvLrEBe3HoTWTyO-HRx0si",
    "rf_model": "18Lyt4gJDpJhdDN0dsJ6s91F0Ef0suvKt",
    "scaler":   "1KrhJWkDhDldkn4PB1FIJcmJiLk5cGV2d",
    "label_encoder": "1ZcEnuPn4gWMvWbWoAHuINVxkwuwM4QWI"
}

GDRIVE_NAMES = {
    "vgg16": "VGG16_best_model.keras",
    "vgg19": "VGG19_best_model.keras",
    "resnet50": "ResNet50_best_model.keras",
    "rf_model": "best_rf_model.joblib",
    "scaler": "crop_scaler.joblib",
    "label_encoder": "crop_label_encoder.joblib"
}

# --- DOWNLOAD FILE IF NOT EXISTS ---
def download_if_missing(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

# --- LOAD MODELS ---
@st.cache_resource
def load_all_models():
    loaded = {}
    for key in GDRIVE_FILES:
        file_id = GDRIVE_FILES[key]
        file_name = GDRIVE_NAMES[key]
        download_if_missing(file_id, file_name)

    try:
        loaded['vgg16'] = tf.keras.models.load_model(GDRIVE_NAMES['vgg16'])
        st.success("✅ VGG16 model loaded.")
    except Exception as e:
        st.error(f"❌ Error loading VGG16: {e}")
        loaded['vgg16'] = None

    try:
        loaded['vgg19'] = tf.keras.models.load_model(GDRIVE_NAMES['vgg19'])
        st.success("✅ VGG19 model loaded.")
    except Exception as e:
        st.error(f"❌ Error loading VGG19: {e}")
        loaded['vgg19'] = None

    try:
        loaded['resnet50'] = tf.keras.models.load_model(GDRIVE_NAMES['resnet50'])
        st.success("✅ ResNet50 model loaded.")
    except Exception as e:
        st.error(f"❌ Error loading ResNet50: {e}")
        loaded['resnet50'] = None

    try:
        loaded['rf_model'] = joblib.load(GDRIVE_NAMES['rf_model'])
        loaded['scaler'] = joblib.load(GDRIVE_NAMES['scaler'])
        loaded['label_encoder'] = joblib.load(GDRIVE_NAMES['label_encoder'])
        st.success("✅ Crop recommendation models loaded.")
    except Exception as e:
        st.error(f"❌ Error loading crop recommendation components: {e}")
        loaded['rf_model'], loaded['scaler'], loaded['label_encoder'] = None, None, None

    return loaded

models = load_all_models()

trained_vgg16 = models.get('vgg16')
trained_vgg19 = models.get('vgg19')
trained_resnet50 = models.get('resnet50')
best_rf_model = models.get('rf_model')
scaler = models.get('scaler')
crop_label_encoder = models.get('label_encoder')

# --- IMPORT PREPROCESSING ---
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input

# --- PREDICTION FUNCTION ---
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
        st.error(f"Error during prediction: {e}")
        return "Prediction Error", 0.0

# --- CROP RECOMMENDATION FUNCTION ---
def recommend_crop(predicted_class_name, rf_model, feature_scaler, crop_le, soil_climate_data):
    if not (rf_model and feature_scaler and crop_le):
        st.warning("⚠️ Crop recommendation models not loaded.")
        return None
    if predicted_class_name in AGRICULTURAL_CLASSES:
        st.subheader("🌾 Crop Recommendation:")
        st.write(f"Detected as **{predicted_class_name}** land.")
        try:
            features = np.array([[soil_climate_data['N'], soil_climate_data['P'], soil_climate_data['K'],
                                  soil_climate_data['temperature'], soil_climate_data['humidity'],
                                  soil_climate_data['ph'], soil_climate_data['rainfall']]])
            features_scaled = feature_scaler.transform(features)
            prediction = rf_model.predict(features_scaled)
            crop = crop_le.inverse_transform(prediction)[0]
            st.success(f"✅ Recommended Crop: **{crop.upper()}**")
        except Exception as e:
            st.error(f"Error in crop recommendation: {e}")
    else:
        st.info(f"'{predicted_class_name}' is not agricultural land.")

# --- STREAMLIT UI ---
st.title("🛰️ Land Cover Classification & Crop Recommendation")
st.markdown("Upload a satellite image and input soil data to get crop recommendations.")

# --- IMAGE UPLOAD ---
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose a satellite image", type=["jpg", "jpeg", "png", "tif"])

# --- SOIL INPUTS ---
st.sidebar.header("Soil & Climate Data")
col1, col2 = st.sidebar.columns(2)
with col1:
    n_val = st.number_input("Nitrogen (N)", 0, 140, 90)
    p_val = st.number_input("Phosphorus (P)", 0, 140, 42)
    k_val = st.number_input("Potassium (K)", 0, 205, 43)
    temp_val = st.number_input("Temperature (°C)", 0.0, 45.0, 20.88)
with col2:
    humidity_val = st.number_input("Humidity (%)", 0.0, 100.0, 82.00)
    ph_val = st.number_input("pH", 0.0, 14.0, 6.5)
    rainfall_val = st.number_input("Rainfall (mm)", 0.0, 300.0, 202.94)

soil_input = {
    'N': n_val, 'P': p_val, 'K': k_val,
    'temperature': temp_val, 'humidity': humidity_val,
    'ph': ph_val, 'rainfall': rainfall_val
}

# --- PREDICTION BUTTON ---
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    if st.button("🧠 Predict & Recommend"):
        st.subheader("📌 Land Cover Classification")
        with st.spinner("Classifying land and recommending crop..."):
            pred16, conf16 = predict_land_cover(uploaded_file, trained_vgg16, vgg16_preprocess_input, EURO_SAT_CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH)
            st.write(f"**VGG16:** {pred16} ({conf16:.2f}%)")
            pred19, conf19 = predict_land_cover(uploaded_file, trained_vgg19, vgg19_preprocess_input, EURO_SAT_CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH)
            st.write(f"**VGG19:** {pred19} ({conf19:.2f}%)")
            pred50, conf50 = predict_land_cover(uploaded_file, trained_resnet50, resnet50_preprocess_input, EURO_SAT_CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH)
            st.write(f"**ResNet50:** {pred50} ({conf50:.2f}%)")

            st.markdown("---")
            recommend_crop(pred50, best_rf_model, scaler, crop_label_encoder, soil_input)
else:
    st.info("Please upload an image to get started.")

st.markdown("---")
st.markdown("🧪 Developed using pre-trained CNN and Random Forest models.")
