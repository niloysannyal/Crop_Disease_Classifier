import streamlit as st
import requests
from PIL import Image
import io

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
# Session state
# =======================
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0



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
        margin-bottom: 40px;
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
    }
    .predict-btn > button:hover {
        background-color: #32CD32 !important; /* Lighter green */
        color: black !important;
    }


    /* Camera title */
    .camera-title {
        font-size: 18px;
        color: white;
        text-align: center;
        margin: 20px 0 5px 0;
    }

    /* Camera logo placeholder (big icon like camera feed) */
    .camera-placeholder {
        display: flex;
        justify-content: center;
        align-items: center;
        border: 2px dashed #888888;  /* default gray border */
        border-radius: 10px;
        background-color: #1E1E1E;  /* background stays dark */
        height: 150px;  /* same as camera feed */
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
    '<div class="subtitle">Upload an image or capture using camera, then select the crop type to predict disease.</div>',
    unsafe_allow_html=True
)

# =======================
# Sidebar Crop Selection
# =======================
crop = st.selectbox("Select Crop", ["Potato", "Corn", "Rice", "Wheat"])

# =======================
# File Uploader
# =======================
uploaded_file = st.file_uploader(
    "Upload Crop Image",
    type=["jpg", "jpeg", "png", "webp"],
    key=f"file_uploader_{st.session_state.uploader_key}"
)

# =======================
# Camera Section
# =======================
st.markdown(
    """
    <div class="camera-title">Take a photo</div>
    <style>
    /* Camera placeholder button styling */
    button[kind="secondary"][title="Click to open camera"], 
    button[key="open_camera"] {
        width: 250px !important;
        height: 250px !important;
        font-size: 140px !important;
        border-radius: 15px !important;
        border: 3px dashed #90EE90 !important;
        background-color: #1E1E1E !important;
        color: #90EE90 !important;
        display: flex;
        justify-content: center;
        align-items: center;
        cursor: pointer;
        margin: 0 auto 20px auto;
        transition: border-color 0.3s;
    }
    button[key="open_camera"]:hover {
        border-color: #32CD32 !important;
        color: #32CD32 !important;
        background-color: #1E1E1E !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if "camera_on" not in st.session_state:
    st.session_state.camera_on = False
if "camera_file" not in st.session_state:
    st.session_state.camera_file = None

camera_container = st.container()

with camera_container:
    if not st.session_state.camera_on:
        # Centered big square camera placeholder
        placeholder_col1, placeholder_col2, placeholder_col3 = st.columns([1, 2, 1])
        with placeholder_col2:
            if st.button("📷", key="open_camera", help="Click to open camera", use_container_width=True):
                st.session_state.camera_on = True
    else:
        # Show camera feed
        st.session_state.camera_file = st.camera_input("Camera Feed")

        # Centered Close Camera button
        close_col1, close_col2, close_col3 = st.columns([1, 1, 1])
        with close_col2:
            if st.button("Close Camera", key="close_cam", use_container_width=True):
                st.session_state.camera_on = False
                st.session_state.camera_file = None




# =======================
# Predict Button
# =======================
st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
if st.button("Predict Disease", use_container_width=True):
    # Get the current camera file and uploaded file
    camera_file = st.session_state.get("camera_file", None)

    # Determine which file to use - uploaded file takes priority if it exists
    file_source = uploaded_file if uploaded_file is not None else camera_file

    if file_source is not None:
        try:
            image = Image.open(file_source).convert("RGB")
            st.image(image, caption="Selected Image", use_container_width=True)

            # Convert image to bytes
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()

            files = {"file": ("image.png", img_bytes, "image/png")}
            data = {"crop": crop.lower()}

            with st.spinner("Predicting..."):
                response = requests.post("http://localhost:8000/predict/", files=files, data=data)

            if response.status_code == 200:
                result = response.json()
                predicted_class = result.get("predicted_class")
                confidence = result.get("confidence")

                st.success(f"Predicted Disease: {predicted_class}")
                st.info(f"Confidence: {confidence}", icon="ℹ️")

                # Centered Try Again button
                try_again_col1, try_again_col2, try_again_col3 = st.columns([1, 1, 1])
                with try_again_col2:
                    if st.button("🔄 Try Again", key="try_again", use_container_width=True):
                        # Clear camera-related session states
                        st.session_state.camera_on = False
                        st.session_state.camera_file = None

                        # Reset file uploader by incrementing its key
                        st.session_state.uploader_key += 1

                        # Force a rerun
                        st.rerun()

            else:
                st.error(f"Error: {response.json().get('error')}")
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

