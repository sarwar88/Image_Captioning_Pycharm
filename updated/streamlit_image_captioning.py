import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# FastAPI endpoint for caption generation
CAPTION_URL = "http://127.0.0.1:8000/generate-caption/"

st.title("Image Captioning Demo")
st.write("Upload an image to generate a caption using greedy or beam search.")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Validate file format
    if uploaded_file.type not in ["image/jpeg", "image/png"]:
        st.error("Invalid file format. Please upload a JPG or PNG image.")
    else:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Select captioning method
        method = st.selectbox("Select captioning method", ["greedy", "beam"])

        # Generate caption button
        if st.button("Generate Caption"):
            with st.spinner("Generating caption..."):
                # Send the image and method to the API
                response = requests.post(
                    CAPTION_URL,
                    files={"file": uploaded_file.getvalue()},
                    data={"method": method},
                )

                # Handle response
                if response.status_code == 200:
                    caption = response.json().get("caption", "No caption generated.")
                    st.success(f"Caption: {caption}")
                else:
                    st.error(f"Error: {response.json().get('error', 'Failed to generate caption.')}")

