import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
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

# =======================
# Load Models
# =======================
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

with st.spinner("Loading crop models..."):
    models_dict = load_all_models()

# =======================
# Prediction Function
# =======================
def predict_disease(crop: str, image_file):
    crop_key = crop.lower()
    if crop_key not in models_dict:
        return None, None
    img = Image.open(image_file).convert("RGB").resize(
        (CROPS[crop_key]["image_size"], CROPS[crop_key]["image_size"])
    )
    img_array = np.expand_dims(np.array(img), axis=0)
    preds = models_dict[crop_key].predict(img_array)
    idx = np.argmax(preds[0])
    class_name = CROPS[crop_key]["class_names"][idx]
    confidence = float(np.max(preds[0])) * 100
    return class_name, confidence

# =======================
# Custom CSS
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
        color: #32CD32;
        text-align: center;
        margin-bottom: 60px;
    }

    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #90EE90;
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

    /* Buttons (Predict + Try Again) */
    div.stButton > button {
        border: 2px solid #444 !important;
        border-radius: 10px;
        background-color: #1E1E1E !important;
        color: white !important;
        font-size: 20px !important;
        height: 50px;
        width: 100% !important;
        margin: 30px 0 30px 0;
    }

    div.stButton > button:hover {
        border-color: #32CD32 !important;
        color: #32CD32 !important;
        background-color: #1E1E1E !important;
    }

    /* Footer */
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
    '<div class="subtitle">Select the crop type, then upload leaf image to predict disease</div>',
    unsafe_allow_html=True
)

# =======================
# Crop Selection
# =======================
crop = st.selectbox("🌾 Select Crop", ["Corn", "Potato", "Rice", "Wheat"])

# =======================
# File Uploader
# =======================
uploaded_file = st.file_uploader(
    "🖼️ Upload Crop Image",
    type=["jpg", "jpeg", "png", "webp"],
    key=f"file_uploader_{st.session_state.uploader_key}"
)


# =========================
# Initialize session state
# =========================
if "predicted_class" not in st.session_state:
    st.session_state.predicted_class = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "camera_file" not in st.session_state:
    st.session_state.camera_file = None
if "prediction_warning" not in st.session_state:
    st.session_state.prediction_warning = None  # store warning message

# =======================
# Determine file source
# =======================
file_source = uploaded_file if uploaded_file is not None else st.session_state.camera_file

# ==========================
# Predict / Try Again logic
# ==========================
def predict_or_reset():
    # Clear previous warning
    st.session_state.prediction_warning = None

    # If prediction exists, reset everything
    if st.session_state.predicted_class is not None:
        st.session_state.predicted_class = None
        st.session_state.confidence = None
        st.session_state.uploader_key += 1
        st.session_state.camera_file = None
    else:
        # Prediction step
        if file_source is not None:
            try:
                image = Image.open(file_source).convert("RGB")
                st.session_state.selected_image = image  # store to session state
                predicted_class, confidence = predict_disease(crop, file_source)
                st.session_state.predicted_class = predicted_class
                st.session_state.confidence = confidence
            except Exception as e:
                st.session_state.prediction_warning = f"Exception: {str(e)}"
        else:
            st.session_state.prediction_warning = "Please upload an image before predicting!"

# ==================================
# Show results if prediction exists
# ==================================
if st.session_state.predicted_class is not None:
    st.image(st.session_state.selected_image, caption="Selected Image", use_container_width=True)
    st.success(f"Predicted Disease: {crop} - {st.session_state.predicted_class}", icon="🧠")
    st.info(f"Confidence: {st.session_state.confidence:.2f}%", icon="🎯")

# ================================================
# Single button handles both predict & try again
# ================================================
btn_label = "🔍 Predict Disease" if st.session_state.predicted_class is None else "🔄 Try Again"

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.button(btn_label, use_container_width=True, on_click=predict_or_reset)

# =======================
# Show warning at bottom
# =======================
if st.session_state.prediction_warning:
    st.warning(st.session_state.prediction_warning)



# =======================
# Footer
# =======================
st.markdown(
    """
    <hr style="border:1px solid #444;">
    <footer style="text-align: center; color: #B0B0B0; font-size: 14px; line-height: 1.6;">
        🌱 Crop Disease Classifier &nbsp;&nbsp;|&nbsp;&nbsp;
        Built by <b>Niloy Sannyal</b> <br>
        Email: <a href="mailto:niloysannyal@gmail.com" style="color:#32CD32;">niloysannyal@gmail.com</a> &nbsp;&nbsp;|&nbsp;&nbsp;
        GitHub: <a href="https://github.com/niloysannyal" target="_blank" style="color:#32CD32;">github.com/niloysannyal</a> <br>
        &copy; 2025 All rights reserved.
    </footer>
    """,
    unsafe_allow_html=True,
)
