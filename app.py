import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image

from model_utils import get_model, get_gradcam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["Benign", "Malignant"]

# transformations identical to those from training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # color variation
    transforms.ToTensor(),
    transforms.Normalize(mean=models.ResNet50_Weights.IMAGENET1K_V1.transforms().mean,
                        std=models.ResNet50_Weights.IMAGENET1K_V1.transforms().std)
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource # load the models only once
def load_models():
    model_pretrained = get_model(pretrained=False, num_classes=2)
    model_scratch = get_model(pretrained=False, num_classes=2)

    model_pretrained.load_state_dict(torch.load("best_pretrained_v4.pth", map_location=device))
    model_scratch.load_state_dict(torch.load("best_scratch_v4.pth", map_location=device))

    model_pretrained.eval()
    model_scratch.eval()
    
    return model_pretrained, model_scratch

##### UI #####
st.title("Breast Cancer Classification")
st.write("Upload image to classify it as benign or malignant")

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg"])

if "history" not in st.session_state:
    st.session_state.history = []

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_tensor = transform(image)
    
    model_pretrained, model_scratch = load_models()
    
    col1, col2 = st.columns(2)
    results = {}
    
    for col, (model_name, model) in zip(
        [col1, col2],
        [("Pretrained", model_pretrained), ("From Scratch", model_scratch)]
    ):
        visualization, img, predicted, probs = get_gradcam(model, image_tensor)
        results[model_name] = (predicted, probs)
        
        with col:
            st.subheader(model_name)
            st.image(image, caption="Original", width='stretch')
            st.write(f"Prediction: {class_names[predicted]}")
            st.write(f"Confidence: {probs[predicted].item():.2%}")
            st.bar_chart({c: p.item() for c, p in zip(class_names, probs)})
            st.image(visualization, caption="GradCAM", width='stretch')
            
    # adding history after classification
    st.session_state.history.append({
        "image": image,
        "pred_pretrained": class_names[results["Pretrained"][0]],
        "confidence_pretrained": results["Pretrained"][1][results["Pretrained"][0]].item(),
        "pred_scratch": class_names[results["From Scratch"][0]],
        "confidence_scratch": results["From Scratch"][1][results["From Scratch"][0]].item()
    })
    
# separated section for history
if st.session_state.history:
    st.divider()
    st.subheader("Classification History")
    
    cols = st.columns(4)
    for idx, entry in enumerate(reversed(st.session_state.history)):
        with cols[idx % 4]:
            st.image(entry["image"], width='stretch')
            st.write(f"**{entry['pred_pretrained']}**")
            st.write(f"{entry['confidence_pretrained']:.0%}")
        
        if idx >= 7:  # afisezi maxim 8 imagini
            break