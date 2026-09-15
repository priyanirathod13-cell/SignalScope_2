import sys
import tempfile
from pathlib import Path

import streamlit as st


# ============================================================
# SignalScope - AI Image Detection
# ============================================================

# Project root
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Make project root available for imports
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from model.predict import predict_image


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="SignalScope",
    page_icon="🔍",
    layout="centered"
)


# ============================================================
# Header
# ============================================================

st.title("🔍 SignalScope")

st.subheader(
    "AI Image Detection & Analysis"
)

st.write(
    "Upload an image to estimate whether it is "
    "AI-generated or real."
)

st.divider()


# ============================================================
# Upload
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ============================================================
# Analyze
# ============================================================

if uploaded_file is not None:

    # Display uploaded image
    st.image(
        uploaded_file,
        caption="Uploaded image",
        width="stretch"
    )

    st.divider()

    if st.button(
        "🔎 Analyze Image",
        type="primary",
        width="stretch"
    ):

        with st.spinner(
            "Analyzing image..."
        ):

            # Save uploaded image temporarily
            suffix = Path(
                uploaded_file.name
            ).suffix

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_path = temp_file.name


            # Run prediction
            result = predict_image(
                temp_path
            )


        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        st.divider()

        label = result["label"]
        confidence = result["confidence"]

        if label == "AI":

            st.error(
                f"🤖 Likely AI-Generated"
            )

        else:

            st.success(
                f"📷 Likely Real"
            )


        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        st.subheader(
            "Prediction Details"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "AI Probability",
                f"{result['ai_probability']:.2f}%"
            )

        with col2:

            st.metric(
                "Real Probability",
                f"{result['real_probability']:.2f}%"
            )


        # ----------------------------------------------------
        # Responsible explanation
        # ----------------------------------------------------

        st.info(
            "The result is a model-based estimate, "
            "not definitive proof of whether an image "
            "is AI-generated."
        )