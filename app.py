import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image
from torchvision import transforms

from models.model_factory import get_model
from weather import get_weather, weather_description


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = (
    "Result/customcnn_standard/checkpoints/best_model.pth"
)

CLASS_NAMES = [
    "Anthracnose",
    "bird eye spot",
    "brown blight",
    "gray light",
    "healthy",
    "red leaf spot",
    "white spot"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Streamlit Page Configuration
# ============================================================

st.set_page_config(
    page_title="Tea Leaf Disease Detection",
    page_icon="🍃",
    layout="wide"
)


# ============================================================
# Load Model
# ============================================================

@st.cache_resource
def load_model():

    model = get_model(
        model_name="customcnn",
        num_classes=len(CLASS_NAMES)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# Image Preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Header
# ============================================================

header_col, weather_col = st.columns(
    [3, 1]
)


# ============================================================
# Application Title
# ============================================================

with header_col:

    st.title("🍃 Tea Leaf Disease Detection")

    st.write(
        "Upload an image of a tea leaf to detect the disease."
    )


# ============================================================
# Small Weather Box
# ============================================================

with weather_col:

    st.markdown(
        """
        <div style="
            padding: 12px;
            border-radius: 12px;
            border: 1px solid #d9d9d9;
            background-color: #f8f9fa;
            margin-top: 10px;
        ">
        <h4 style="
            margin-top: 0;
            margin-bottom: 10px;
        ">
        🌦️ Weather
        </h4>
        """,
        unsafe_allow_html=True
    )

    # Default location
    latitude = 10.8505
    longitude = 76.2711

    try:

        weather = get_weather(
            latitude,
            longitude
        )

        temperature = weather[
            "temperature_2m"
        ]

        humidity = weather[
            "relative_humidity_2m"
        ]

        precipitation = weather[
            "precipitation"
        ]

        weather_code = weather[
            "weather_code"
        ]

        description = weather_description(
            weather_code
        )

        st.markdown(
            f"""
            <div style="
                font-size: 13px;
                line-height: 1.7;
            ">
            🌡️ <b>{temperature} °C</b><br>
            💧 Humidity: <b>{humidity}%</b><br>
            🌧️ Rain: <b>{precipitation} mm</b><br>
            ☁️ {description}
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception:

        st.warning(
            "Weather unavailable"
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# Divider
# ============================================================

st.divider()


# ============================================================
# Image Upload
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Choose a tea leaf image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# Image Processing
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image_col, details_col = st.columns(
        [1, 1]
    )


    # --------------------------------------------------------
    # Display Image
    # --------------------------------------------------------

    with image_col:

        st.image(
            image,
            caption="Uploaded Tea Leaf",
            use_container_width=True
        )


    # --------------------------------------------------------
    # Prediction Button
    # --------------------------------------------------------

    with details_col:

        st.subheader(
            "🔍 Disease Detection"
        )

        st.write(
            "Click the button below to analyze "
            "the uploaded tea leaf."
        )

        predict_button = st.button(
            "🔍 Predict Disease",
            use_container_width=True
        )


    # ========================================================
    # Prediction
    # ========================================================

    if predict_button:

        try:

            # ------------------------------------------------
            # Load Model
            # ------------------------------------------------

            model = load_model()


            # ------------------------------------------------
            # Preprocess Image
            # ------------------------------------------------

            input_tensor = transform(
                image
            )

            input_tensor = input_tensor.unsqueeze(
                0
            )

            input_tensor = input_tensor.to(
                DEVICE
            )


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            with torch.no_grad():

                output = model(
                    input_tensor
                )

                probabilities = F.softmax(
                    output,
                    dim=1
                )


            # ------------------------------------------------
            # Get Predicted Class
            # ------------------------------------------------

            predicted_index = torch.argmax(
                probabilities,
                dim=1
            ).item()

            predicted_class = CLASS_NAMES[
                predicted_index
            ]

            confidence = (
                probabilities[0][
                    predicted_index
                ].item()
                * 100
            )


            # ------------------------------------------------
            # Display Result
            # ------------------------------------------------

            st.divider()

            st.header(
                "🧠 Prediction Result"
            )


            result_col1, result_col2 = st.columns(
                2
            )


            with result_col1:

                if predicted_class.lower() == "healthy":

                    st.success(
                        f"🌿 {predicted_class}"
                    )

                else:

                    st.error(
                        f"🌿 {predicted_class}"
                    )


            with result_col2:

                st.info(
                    f"🎯 Confidence: "
                    f"{confidence:.2f}%"
                )


            # ------------------------------------------------
            # Prediction Probabilities
            # ------------------------------------------------

            st.subheader(
                "📊 Prediction Probabilities"
            )

            for i, class_name in enumerate(
                CLASS_NAMES
            ):

                probability = probabilities[
                    0
                ][i].item()

                percentage = (
                    probability * 100
                )

                st.write(
                    f"**{class_name}**: "
                    f"{percentage:.2f}%"
                )

                st.progress(
                    probability
                )


        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "🍃 Tea Leaf Disease Detection System | "
    "CustomCNN + Weather Information"
)
