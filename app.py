import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path

from model import ImprovedCNN


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Bloomora | Flower Recognition",
    page_icon="🌷",
    layout="wide"
)


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

.stApp {
    background-color: #F7F8F3;
    color: #344638;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #344638 !important;
}

p, label, span {
    font-family: 'DM Sans', sans-serif;
}

.hero {
    background: linear-gradient(120deg, #E6EDDF, #F1EBDD);
    padding: 55px 45px;
    border-radius: 22px;
    margin-bottom: 35px;
    border: 1px solid #DCE5D6;
}

.hero-tag {
    color: #71866A;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 18px;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    color: #344638;
    font-size: 48px;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 18px;
}

.hero-description {
    color: #68776A;
    font-size: 15px;
    line-height: 1.8;
    max-width: 650px;
}

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 30px;
    font-weight: 700;
    color: #344638;
    margin-top: 15px;
    margin-bottom: 5px;
}

.section-description {
    color: #7A897B;
    font-size: 14px;
    margin-bottom: 22px;
}

div[data-testid="stFileUploader"] {
    background-color: #FFFFFF;
    border: 1px dashed #A9B99F;
    border-radius: 15px;
    padding: 20px;
}

.stButton > button {
    background-color: #71866A;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 25px;
    font-weight: 600;
    transition: 0.2s;
    width: 100%;
}

.stButton > button:hover {
    background-color: #536B50;
    color: white;
}

div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8DC;
    padding: 18px;
    border-radius: 14px;
}

div[data-testid="stMetricValue"] {
    color: #536B50;
}

div[data-testid="stProgressBar"] > div {
    background-color: #71866A;
}

div[data-testid="stExpander"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8DC;
    border-radius: 12px;
}

.catalog-name {
    font-family: 'Playfair Display', serif;
    color: #344638;
    font-size: 21px;
    font-weight: 600;
}

.catalog-description {
    color: #778477;
    font-size: 13px;
    line-height: 1.6;
}

.footer {
    text-align: center;
    color: #879487;
    font-size: 12px;
    padding-top: 30px;
    padding-bottom: 10px;
}

div[data-testid="stImage"] img {
    border-radius: 14px;
}

hr {
    border-color: #E1E7DC;
}

@media (max-width: 768px) {
    .hero {
        padding: 35px 25px;
    }

    .hero-title {
        font-size: 34px;
    }
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. MODEL CONFIGURATION
# ==========================================

IMAGE_SIZE = 128

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = Path(__file__).parent / "best_cnn_improved.pth"


# ==========================================
# 4. LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    class_names = checkpoint["classes"]

    model = ImprovedCNN(
        num_classes=len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.to(device)
    model.eval()

    return model, class_names


model, CLASS_NAMES = load_model()


# ==========================================
# 5. IMAGE PREPROCESSING
# ==========================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ==========================================
# 6. FLOWER CATALOG DATA
# ==========================================

flower_catalog = [

    {
        "name": "Chrysanthemum",
        "scientific": "Chrysanthemum",
        "emoji": "🌼",
        "description": "A beautiful flower known for its layered petals and vibrant colors.",
        "sunlight": "5–6 hours of sunlight daily.",
        "watering": "Keep the soil consistently moist but never waterlogged.",
        "care": "Remove wilted flowers regularly and provide well-draining soil.",
        "color": "White, yellow, pink, purple and red"
    },

    {
        "name": "Hibiscus",
        "scientific": "Hibiscus rosa-sinensis",
        "emoji": "🌺",
        "description": "A tropical flower recognized by its large petals and vibrant appearance.",
        "sunlight": "Around 6 hours or more of direct sunlight.",
        "watering": "Water regularly, especially during hot weather.",
        "care": "Prune dead branches and fertilize periodically during the growing season.",
        "color": "Red, pink, yellow, orange and white"
    },

    {
        "name": "Marigold",
        "scientific": "Tagetes",
        "emoji": "🌼",
        "description": "A cheerful flower with bright yellow and orange petals.",
        "sunlight": "Full sunlight, preferably 6 hours or more.",
        "watering": "Water when the top layer of soil begins to dry.",
        "care": "Remove faded flowers and avoid overwatering.",
        "color": "Yellow, orange and golden"
    },

    {
        "name": "Petunia",
        "scientific": "Petunia",
        "emoji": "🌸",
        "description": "A colorful trumpet-shaped flower available in many varieties.",
        "sunlight": "Around 5–6 hours of sunlight daily.",
        "watering": "Water regularly, especially when planted in containers.",
        "care": "Remove faded flowers and use well-draining soil.",
        "color": "Purple, pink, white, red and blue"
    },

    {
        "name": "Rose",
        "scientific": "Rosa",
        "emoji": "🌹",
        "description": "A popular flowering plant admired for its beauty and fragrance.",
        "sunlight": "Around 6 hours of direct sunlight daily.",
        "watering": "Water the soil directly and avoid keeping leaves wet.",
        "care": "Prune dead branches and provide nutrient-rich, well-draining soil.",
        "color": "Red, pink, white, yellow and orange"
    }

]


# ==========================================
# 7. HERO SECTION
# ==========================================

st.markdown("""
<div class="hero">

    <div class="hero-tag">
        🌷 BLOOMORA · AI-POWERED FLOWER CLASSIFICATION
    </div>

    <div class="hero-title">
        Discover the Beauty<br>
        of Nature.
    </div>

    <div class="hero-description">
        Explore the beauty of flowers through artificial intelligence.
        Simply upload a flower image and let Bloomora identify its
        category using deep learning technology.
    </div>

</div>
""", unsafe_allow_html=True)


# ==========================================
# 8. FLOWER IDENTIFICATION SECTION
# ==========================================

st.markdown(
    '<div class="section-title">Identify Your Flower</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Upload a flower image and let Bloomora discover its identity.'
    '</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(
    [1, 1],
    gap="large"
)

with col1:

    st.markdown("### 📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose a flower image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a flower."
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.success("Image uploaded successfully!")

    else:

        st.info(
            "Please upload a flower image to start identification."
        )


with col2:

    st.markdown("### 🖼️ Image Preview")

    if uploaded_file is not None:

        st.image(
            image,
            caption="Your uploaded flower image",
            use_container_width=True
        )

    else:

        st.markdown("""
        <div style="
            background-color: #EDF1E8;
            border: 1px dashed #A9B99F;
            border-radius: 15px;
            padding: 70px 20px;
            text-align: center;
            color: #7A897B;
        ">
            <div style="font-size: 45px;">🌷</div>
            <p>Your flower image will appear here.</p>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# 9. PREDICTION SECTION
# ==========================================

if uploaded_file is not None:

    st.write("")

    predict_button = st.button(
        "🌷 Identify Flower",
        use_container_width=True
    )

    if predict_button:

        with st.spinner("Bloomora is identifying your flower..."):

            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():

                outputs = model(input_tensor)

                probabilities = F.softmax(
                    outputs,
                    dim=1
                )[0]

                confidence, predicted_idx = torch.max(
                    probabilities,
                    dim=0
                )

            predicted_class = CLASS_NAMES[predicted_idx.item()]

            st.session_state["prediction"] = {
                "class": predicted_class,
                "confidence": confidence.item() * 100,
                "probabilities": probabilities.cpu().tolist()
            }

            st.session_state["prediction_file"] = uploaded_file.getvalue()


    # Display prediction for the current uploaded image

    if (
        "prediction" in st.session_state
        and st.session_state.get("prediction_file") == uploaded_file.getvalue()
    ):

        result = st.session_state["prediction"]

        st.write("")
        st.divider()

        st.markdown(
            '<div class="section-title">Prediction Result</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-description">'
            'Here is the flower identified by our AI model.'
            '</div>',
            unsafe_allow_html=True
        )

        result_col1, result_col2 = st.columns(
            [1, 1],
            gap="large"
        )

        with result_col1:

            st.markdown(
                f"""
                <div style="
                    background-color: #E8EFE3;
                    border: 1px solid #D4E1CE;
                    border-radius: 18px;
                    padding: 30px;
                    text-align: center;
                ">
                    <div style="font-size: 42px;">🌸</div>
                    <p style="
                        color: #71866A;
                        font-size: 12px;
                        font-weight: 700;
                        letter-spacing: 2px;
                    ">
                        IDENTIFIED FLOWER
                    </p>
                    <h2 style="
                        color: #344638;
                        font-family: 'Playfair Display', serif;
                    ">
                        {result["class"]}
                    </h2>
                    <p style="color: #71866A;">
                        Bloomora Prediction
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with result_col2:

            st.metric(
                label="Prediction Confidence",
                value=f'{result["confidence"]:.2f}%'
            )

            st.write("")

            st.markdown("**Class Probabilities**")

            for i, class_name in enumerate(CLASS_NAMES):

                probability = (
                    result["probabilities"][i] * 100
                )

                st.write(
                    f"**{class_name}** — {probability:.2f}%"
                )

                st.progress(
                    min(probability / 100, 1.0)
                )

        st.caption(
            "Note: Prediction confidence represents the model's "
            "estimated probability, not a guarantee of correctness."
        )


# ==========================================
# 10. FLOWER CATALOG SECTION
# ==========================================

st.write("")
st.divider()

st.markdown(
    '<div class="section-title">🌷 Flower Catalog</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Explore our flower collection and discover how to care for them.'
    '</div>',
    unsafe_allow_html=True
)


catalog_cols = st.columns(2, gap="large")

for i, flower in enumerate(flower_catalog):

    with catalog_cols[i % 2]:

        with st.container(border=True):

            st.markdown(
                f"""
                <div class="catalog-name">
                    {flower["emoji"]} {flower["name"]}
                </div>
                <p class="catalog-description">
                    <i>{flower["scientific"]}</i>
                </p>
                <p>{flower["description"]}</p>
                """,
                unsafe_allow_html=True
            )

            with st.expander("🌱 View Care Guide"):

                st.markdown("**🎨 Flower Colors**")
                st.write(flower["color"])

                st.markdown("**☀️ Sunlight**")
                st.write(flower["sunlight"])

                st.markdown("**💧 Watering**")
                st.write(flower["watering"])

                st.markdown("**🌿 Care Tips**")
                st.write(flower["care"])


# ==========================================
# 11. ABOUT BLOOMORA SECTION
# ==========================================

st.write("")
st.divider()

st.markdown(
    '<div class="section-title">About Bloomora</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Discover the technology behind our flower classification system.'
    '</div>',
    unsafe_allow_html=True
)

about_col1, about_col2, about_col3 = st.columns(3)

with about_col1:

    with st.container(border=True):

        st.markdown("### 🧠 Deep Learning")

        st.write(
            "Powered by a Convolutional Neural Network (CNN) "
            "trained to recognize flower images."
        )

with about_col2:

    with st.container(border=True):

        st.markdown("### 🌸 5 Flower Classes")

        st.write(
            "Recognizes Chrysanthemum, Hibiscus, Marigold, "
            "Petunia, and Rose."
        )

with about_col3:

    with st.container(border=True):

        st.markdown("### 🌱 Flower Care")

        st.write(
            "Provides basic information and care recommendations "
            "to help users learn about different flowers."
        )


# ==========================================
# 12. DISCLAIMER
# ==========================================

st.write("")

st.info(
    "🌿 **Disclaimer:** Bloomora is an educational flower "
    "classification application. Its predictions and care "
    "recommendations are intended for informational purposes "
    "only and may not always be accurate."
)


# ==========================================
# 13. FOOTER
# ==========================================

st.markdown("""
<div class="footer">

    <p>🌷 Bloomora · Discover the Beauty of Nature</p>

    <p>
        Powered by Deep Learning and Convolutional Neural Networks
    </p>

    <p>© 2026 Bloomora. All rights reserved.</p>

</div>
""", unsafe_allow_html=True)