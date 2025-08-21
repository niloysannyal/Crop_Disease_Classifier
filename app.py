import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os

# =======================
# Streamlit Page Config
# =======================
st.set_page_config(
    page_title="Crop Disease Classifier",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded",
)


# =======================
# Session State
# =======================
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False
if "camera_file" not in st.session_state:
    st.session_state.camera_file = None



# =======================
# Crop models info
# =======================
CROPS = {
    "corn": {
        "model_path": "./corn/saved_models/corn_disease_model.h5",
        "class_names": ["Common Rust", "Gray Leaf Spot", "Healthy", "Northern Leaf Blight"],
        "image_size": 224
    },
    "potato": {
        "model_path": "./potato/saved_models/potato_disease_model.h5",
        "class_names": ["Early Blight", "Healthy", "Late Blight"],
        "image_size": 224
    },
    "rice": {
        "model_path": "./rice/saved_models/rice_disease_model.h5",
        "class_names": ["Brown Spot", "Healthy", "Leaf Blast", "Neck Blast"],
        "image_size": 224
    },
    "wheat": {
        "model_path": "./wheat/saved_models/wheat_disease_model.h5",
        "class_names": ["Brown Rust", "Healthy", "Yellow Rust"],
        "image_size": 224
    },
}



@st.cache_resource
def load_all_models():
    models = {}
    for crop_name, info in CROPS.items():
        if os.path.exists(info["model_path"]):
            models[crop_name] = tf.keras.models.load_model(info["model_path"], compile=False)
            print(f"Loaded {crop_name} model.")
        else:
            print(f"Model for {crop_name} not found at {info['model_path']}")
    return models

models_dict = load_all_models()


# =======================
# Prediction Function
# =======================
def predict_disease(crop: str, image_file):
    crop_key = crop.lower()
    if crop_key not in models_dict:
        return None, None
    image = Image.open(image_file).convert("RGB")
    img_array = image.resize((CROPS[crop_key]["image_size"], CROPS[crop_key]["image_size"]))
    img_array = np.expand_dims(np.array(img_array), axis=0)
    preds = models_dict[crop_key].predict(img_array)
    idx = np.argmax(preds[0])
    class_name = CROPS[crop_key]["class_names"][idx]
    confidence = float(np.max(preds[0]))
    confidence = confidence*100
    return class_name, confidence



# =======================
# Custom CSS for Dark Mode + Layout
# =======================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #121212;
        color: #E0E0E0;
    }

    .title {
        font-size: 42px;
        font-weight: bold;
        color: #90EE90;
        text-align: center;
        margin-bottom: 60px;
    }

    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #B0B0B0;
        margin-bottom: 20px;
    }

    /* File uploader */
    .stFileUploader>div>div {
        border: 2px dashed #90EE90;
        border-radius: 10px;
        padding: 15px;
        background-color: #1E1E1E;
    }

    /* Selectbox label */
    .stSelectbox label {
        font-size: 20px !important;
        font-weight: 600;
    }
    .stSelectbox div[data-baseweb="select"] {
        font-size: 18px !important;
        height: 50px;
        border-radius: 8px;
    }

    /* Prediction button */
    .predict-btn > button {
        background-color: #228B22 !important; /* Forest green */
        color: white !important;
        font-size: 20px !important;
        height: 55px;
        width: 100% !important;
        border-radius: 10px;
        border: none;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .predict-btn > button:hover {
        background-color: #32CD32 !important; /* Lighter green */
        color: black !important;
    }


    /* Camera title */
    .camera-title {
        font-size: 16px;
        color: white;
        text-align: center;
        margin: 15px 0 5px 0;
    }

    /* Camera logo placeholder (big icon like camera feed) */
    .camera-placeholder {
        display: flex;
        justify-content: center;
        align-items: center;
        border: 2px dashed #888888;
        border-radius: 10px;
        background-color: #1E1E1E;
        height: 150px;
        font-size: 80px;
        cursor: pointer;
        color: #90EE90;
        margin-bottom: 20px;
        transition: border-color 0.3s;
    }
    
    .camera-placeholder:hover {
        border-color: #90EE90; /* only border turns green */
        background-color: #1E1E1E; /* keep background same */
        color: #90EE90;
    }


    /* Center buttons (for camera buttons only) */
    .center-btn {
        display: flex;
        justify-content: center;
        margin-top: 15px;
    }
    .center-btn button {
        width: 50% !important;
    }

    footer {
        color: white;
        font-size: 16px;
        text-align: center;
        margin-top: 50px;
    }
    footer b {
        font-weight: 900;
        color: white;
    }

    hr { border: 1px solid #444; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =======================
# App Title + Subtitle
# =======================
st.markdown('<div class="title">🌱 Crop Disease Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Select the crop type, then upload or capture leaf image to predict disease.</div>',
    unsafe_allow_html=True
)

# =======================
# Sidebar Crop Selection
# =======================
crop = st.selectbox("🌾 Select Crop", ["Corn", "Potato",  "Rice", "Wheat"])

# =======================
# File Uploader
# =======================
uploaded_file = st.file_uploader(
    "🖼️ Upload Crop Image",
    type=["jpg", "jpeg", "png", "webp"],
    key=f"file_uploader_{st.session_state.uploader_key}"
)

# =======================
# Camera Section
# =======================
st.markdown(
    """
    <div class="camera-title">Capture a photo</div>
    """,
    unsafe_allow_html=True
)

camera_container = st.container()

# Define functions to toggle camera
def open_camera():
    st.session_state.camera_on = True

def close_camera():
    st.session_state.camera_on = False
    st.session_state.camera_file = None

with camera_container:
    if not st.session_state.camera_on:
        # Show camera placeholder button
        placeholder_col1, placeholder_col2, placeholder_col3 = st.columns([2, 1, 2])
        with placeholder_col2:
            st.button("📷", key="open_camera", help="Click to open camera",
                      use_container_width=True, on_click=open_camera)
    else:
        # Show camera feed
        st.session_state.camera_file = st.camera_input("Camera Feed")

        # Close Camera button
        close_col1, close_col2, close_col3 = st.columns([2, 1, 2])
        with close_col2:
            st.button("Close Camera", key="close_cam", use_container_width=True, on_click=close_camera)



# =======================
# Predict Button
# =======================
st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
if st.button("🔍 Predict Disease", use_container_width=True):
    # Get the current camera file and uploaded file
    camera_file = st.session_state.get("camera_file", None)

    # Determine which file to use - uploaded file takes priority if it exists
    file_source = uploaded_file if uploaded_file is not None else camera_file

    if file_source is not None:
        try:
            image = Image.open(file_source).convert("RGB")
            st.image(image, caption="Selected Image", use_container_width=True)

            predicted_class, confidence = predict_disease(crop, file_source)

            if predicted_class:
                st.success(f"Predicted Disease:  {predicted_class}", icon="🧠")
                st.info(f"Confidence:  {confidence:.2f}%", icon="🎯")

                # Centered Try Again button
                try_again_col1, try_again_col2, try_again_col3 = st.columns([1, 1, 1])
                with try_again_col2:
                    if st.button("🔄 Try Again", key="try_again", use_container_width=True):
                        st.session_state.camera_on = False
                        st.session_state.camera_file = None
                        st.session_state.uploader_key += 1
                        st.rerun()

            else:
                st.error(f"No model loaded for {crop}!")
        except Exception as e:
            st.error(f"Exception: {str(e)}")
    else:
        st.warning("Please upload or capture an image before predicting!")
st.markdown('</div>', unsafe_allow_html=True)




# =======================
# Professional Footer
# =======================
st.markdown(
    """
    <hr style="border:1px solid #444;">
    <footer style="text-align: center; color: #B0B0B0; font-size: 14px; line-height: 1.6;">
        🌱 Crop Disease Classifier &nbsp;&nbsp;|&nbsp;&nbsp;
        Built by <b>Niloy Sannyal</b> <br>
        Email: <a href="mailto:niloysannyal@gmail.com" style="color:#90EE90;">niloysannyal@gmail.com</a> &nbsp;&nbsp;|&nbsp;&nbsp;
        GitHub: <a href="https://github.com/niloysannyal" target="_blank" style="color:#90EE90;">github.com/niloysannyal</a> <br>
        &copy; 2025 All rights reserved.
    </footer>
    """,
    unsafe_allow_html=True,
)

