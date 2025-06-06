# Land & Crop Recommendation App

This is a Streamlit app that classifies land cover from satellite images and recommends crops based on soil and climate data.

## How to Use

1. Clone or download this repository (code only).

2. Download the trained model files from Google Drive:  
[Models Download Link](https://drive.google.com/drive/folders/1Mg6gARHn2ZXtuYaYMeniVlVqTY7hii9E?usp=sharing)

3. Place the downloaded model files in a folder and update the `DRIVE_MODEL_PATH` in `app.py` to point to that folder.

4. Install required packages:  
pip install -r requirements.txt

markdown
Copy
Edit

5. Run the app:  
streamlit run app.py

yaml
Copy
Edit

6. Upload a satellite image and enter soil/climate data to get land cover classification and crop recommendations.

## Notes

- Models are **not included** in this repo due to file size.
- Ensure you have access to the Google Drive folder to download the models.

---

Developed by Bhavika Reddy Alsani