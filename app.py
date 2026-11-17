
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

from model import ImprovedCNN

# Configuration
IMAGE_SIZE = 128

CLASS_NAMES = [
    "Chrysanthemum",
    "Hibiscus",
    "Marigold",
    "Petunia",
    "Rose"
]

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Model path
MODEL_PATH = Path(__file__).parent / "best_cnn.pth"

# Load model
@st.cache_resource
def load_model():
    model = ImprovedCNN(num_classes=len(CLASS_NAMES))

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model

model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Streamlit UI
st.title("Deep Learning Image Classification")

st.write(
    "Upload an image of a flower to predict its class."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    input_tensor = transform(image).unsqueeze(0).to(device)

    if st.button("Predict"):

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)

            confidence, predicted = torch.max(
                probabilities,
                dim=1
            )

        predicted_class = CLASS_NAMES[predicted.item()]

        st.success(f"Prediction: {predicted_class}")

        st.write(
            f"Confidence: {confidence.item() * 100:.2f}%"
        )

        st.subheader("Class Probabilities")

        for i, class_name in enumerate(CLASS_NAMES):
            st.write(
                f"{class_name}: "
                f"{probabilities[0][i].item() * 100:.2f}%"
            )