import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model once at the start of the app
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_model.keras")

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = load_model()
    image = Image.open(test_image)
    image = image.resize((256, 256))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease and Health Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE AND HEALTH RECOGNITION SYSTEM")
    image_path = "plant.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition and Health Monitoring System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets while preserving the directory structure.
    A new directory containing 33 test images is created later for prediction purposes.

    #### Content
    1. train (70,295 images)
    2. test (33 images)
    3. validation (17,572 images)
    """)

# Prediction Page
elif app_mode == "Disease and Health Recognition":
    st.header("Disease and Health Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])
    
    if test_image:
        st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Analyzing image..."):
                try:
                    result_index = model_prediction(test_image)
                    class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                                   'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                                   'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                                   'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                                   'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                                   'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                                   'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                                   'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                                   'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                                   'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                                   'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                                   'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                                   'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                                   'Tomato___healthy']
                    
                    predicted_class = class_names[result_index]
                    
                    # Dictionary of health suggestions
                    health_suggestions = {
                        'Apple___Apple_scab': "Prune infected leaves, apply fungicide, and ensure proper air circulation around trees.",
    'Apple___Black_rot': "Remove infected fruit and branches, use fungicide, and keep the orchard clean.",
    'Apple___Cedar_apple_rust': "Remove nearby cedar trees, use rust-resistant varieties, and apply fungicide.",
    'Apple___healthy': "No issues detected. Maintain regular watering and fertilization.",
    'Blueberry___healthy': "No issues detected. Ensure soil acidity is appropriate and provide adequate water.",
    'Cherry_(including_sour)___Powdery_mildew': "Use fungicide, avoid overhead watering, and prune affected areas.",
    'Cherry_(including_sour)___healthy': "No issues detected. Continue with regular care and monitoring.",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Use fungicide, rotate crops, and remove infected debris.",
    'Corn_(maize)___Common_rust_': "Apply fungicide and use rust-resistant varieties.",
    'Corn_(maize)___Northern_Leaf_Blight': "Use resistant hybrids, rotate crops, and apply fungicide if necessary.",
    'Corn_(maize)___healthy': "No issues detected. Maintain regular care for optimal growth.",
    'Grape___Black_rot': "Remove and destroy infected leaves and fruit, and apply fungicide.",
    'Grape___Esca_(Black_Measles)': "Prune infected vines, ensure proper irrigation, and avoid over-fertilizing.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Use fungicide, prune affected areas, and ensure good air circulation.",
    'Grape___healthy': "No issues detected. Regular monitoring and care are recommended.",
    'Orange___Haunglongbing_(Citrus_greening)': "Remove infected trees, use resistant rootstocks, and control insect vectors.",
    'Peach___Bacterial_spot': "Apply copper-based bactericides, prune infected areas, and avoid overhead irrigation.",
    'Peach___healthy': "No issues detected. Continue with regular watering and care.",
    'Pepper,_bell___Bacterial_spot': "Use copper-based fungicide, rotate crops, and remove infected plants.",
    'Pepper,_bell___healthy': "No issues detected. Maintain consistent watering and nutrient supply.",
    'Potato___Early_blight': "Use fungicide, rotate crops, and remove plant debris after harvest.",
    'Potato___Late_blight': "Apply fungicide and practice crop rotation to minimize disease spread.",
    'Potato___healthy': "No issues detected. Ensure proper soil health and pest control.",
    'Raspberry___healthy': "No issues detected. Keep monitoring for pests and diseases.",
    'Soybean___healthy': "No issues detected. Regular crop rotation and pest monitoring are essential.",
    'Squash___Powdery_mildew': "Use fungicide, water at the base of the plant, and ensure good air circulation.",
    'Strawberry___Leaf_scorch': "Remove infected leaves, avoid overhead watering, and apply fungicide.",
    'Strawberry___healthy': "No issues detected. Continue regular monitoring and care.",
    'Tomato___Bacterial_spot': "Use copper-based sprays, avoid wetting leaves, and remove infected plants.",
    'Tomato___Early_blight': "Apply fungicide, remove affected leaves, and mulch plants to prevent soil splashing.",
    'Tomato___Late_blight': "Remove infected plants immediately, use fungicide, and avoid overhead irrigation.",
    'Tomato___Leaf_Mold': "Improve air circulation, use resistant varieties, and apply fungicide if necessary.",
    'Tomato___Septoria_leaf_spot': "Prune affected leaves, avoid overhead watering, and use fungicide.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Use insecticidal soap, keep plants well-watered, and encourage natural predators.",
    'Tomato___Target_Spot': "Apply fungicide and remove diseased leaves to prevent spread.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Remove infected plants and control whiteflies with insecticidal soap.",
    'Tomato___Tomato_mosaic_virus': "Remove infected plants, disinfect tools, and avoid smoking near plants.",
    'Tomato___healthy': "No issues detected. Continue regular care and monitoring."
                    }
                    
                    health_suggestion = health_suggestions.get(predicted_class, "No suggestion available.")
                    
                    # Display the predicted disease and health suggestion
                    st.success(f"Model Prediction: {predicted_class}")
                    st.info(f"Health Suggestion: {health_suggestion}")
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
