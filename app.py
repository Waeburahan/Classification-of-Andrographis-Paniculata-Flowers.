import streamlit as st
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os


# Define class labels
class_labels = ['Bloom', 'Bud', 'Full_bloom', 'Half_percent_bloom', 'Withered']

# Load the pre-trained model
@st.cache_resource
def load_trained_model():
    model_path = os.path.join(os.getcwd(), "mobilenetv3_large_augment_checkpoint_fold4.pt")  # Ensure correct path to the model file
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the file is present.")
    
    # Load the model on CPU
    model = torch.load(model_path, map_location=torch.device('cpu'))  # Force loading on CPU
    model.half()  # Convert model to use half-precision if necessary
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess the uploaded image
def preprocess_image(image):
    target_size = (240, 240)  # Adjust based on the model's expected input size
    
    # Ensure the image is in RGB format (3 channels)
    image = image.convert('RGB')
    
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization based on ImageNet stats (if relevant)
    ])
    image_tensor = preprocess(image)
    return image_tensor.unsqueeze(0)  # Add batch dimension

# Display the app title and description
st.title("Classification of Andrographis Paniculata Flowers. ")
st.write("Upload an image to classify.")

# Load the model
try:
    model = load_trained_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Convert the input tensor to float16 if necessary
        preprocessed_image = preprocessed_image.to(torch.float16)

        # Make prediction
        with torch.no_grad():  # Disable gradient computation
            output = model(preprocessed_image)

        # Get the prediction probabilities (softmax)
        probabilities = torch.nn.functional.softmax(output, dim=1)

        # Display the class probabilities (confidence)
        st.write("### Prediction Probabilities")
        for i, prob in enumerate(probabilities[0]):
            st.write(f"{class_labels[i]}: {prob.item() * 100:.2f}%")

        # Get the predicted class
        _, predicted_class = torch.max(output, 1)
        predicted_class_name = class_labels[predicted_class.item()]
        st.write(f"### Predicted Class")
        st.write(f"Predicted class: {predicted_class_name}")
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.write("Please upload an image to get started.")
