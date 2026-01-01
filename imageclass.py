import streamlit as st
import torch

# Configure Streamlit page
st.set_page_config(
    page_title="CPU-Based Image Classification",
    layout="centered"
)

# Force CPU usage
device = torch.device("cpu")

from torchvision import models, transforms
from PIL import Image
import pandas as pd
import torch.nn.functional as F

# Load pre-trained ResNet18 model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = model.to(device)
model.eval()

# Image preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load ImageNet class labels
labels = models.ResNet18_Weights.DEFAULT.meta["categories"]

st.title("🖼️ Image Classification using ResNet18 (CPU)")
st.write("Upload an image (JPG or PNG) to classify it.")

uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

    probabilities = F.softmax(outputs, dim=1)[0]
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    results = []
    for i in range(5):
        results.append({
            "Class": labels[top5_catid[i]],
            "Probability (%)": round(top5_prob[i].item() * 100, 2)
        })

    df = pd.DataFrame(results)
    st.subheader("🔍 Top-5 Predictions")
    st.table(df)

    st.subheader("📊 Prediction Probability Distribution")
    st.bar_chart(
        data=df.set_index("Class")["Probability (%)"]
    )
