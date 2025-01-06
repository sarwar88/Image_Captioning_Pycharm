from io import BytesIO

import joblib
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image
from tqdm import tqdm

# Initialize FastAPI app
app = FastAPI()

# Load pre-trained models and required data at startup
print("Loading models and vocabulary...")
try:
    resnet_model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", pooling="avg")
    print("ResNet50 model loaded.")

    words_to_indices = joblib.load("../sobject/words_to_indices.joblib")
    indices_to_words = joblib.load("../sobject/indices_to_words.joblib")
    lstm_model = joblib.load("../sobject/image_captioning_model.joblib")

    print("Serialized objects loaded successfully.")

    max_length = 40
    print("Models and vocabulary loaded successfully.")

except Exception as e:
    print(f"Error loading models or vocabulary: {e}")
    raise HTTPException(status_code=500, detail="Error loading models or vocabulary.")


@app.post("/generate-caption/")
async def generate_caption(file: UploadFile, method: str = Form("greedy")):
    """
    Generate a caption for the uploaded image using the specified method (greedy or beam search).
    """
    try:
        print("Received a request to generate caption.")

        # Validate file type
        print("File format valid.")
        # Read and preprocess the image
        print("Reading and preprocessing image...")
        img = Image.open(BytesIO(await file.read())).convert("RGB")
        img = img.resize((224, 224))  # Resize to ResNet50 input size
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        print("Image preprocessed.")

        # Extract features using ResNet50
        print("Extracting features using ResNet50 model...")
        photo_features = resnet_model.predict(img_array).squeeze()
        print("Feature extraction complete.")

        # Generate caption based on the method
        if method == "greedy":
            print("Using greedy search method.")
            caption = greedy_search(photo_features)
        elif method == "beam":
            print("Using beam search method.")
            caption = beam_search(photo_features, k=3)
        else:
            print(f"Invalid method selected: {method}")
            raise HTTPException(status_code=400, detail="Invalid method. Choose 'greedy' or 'beam'.")

        print("Caption generated successfully.")
        return JSONResponse(content={"caption": " ".join(caption)})

    except Exception as e:
        print(f"Error generating caption: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions
def greedy_search(photo):
    """Generate a caption using greedy search."""
    print("Starting greedy search...")
    photo = photo.reshape(1, 2048)
    in_text = "<start>"
    for _ in range(max_length):
        sequence = [words_to_indices[s] for s in in_text.split() if s in words_to_indices]
        sequence = pad_sequences([sequence], maxlen=max_length, padding="post")
        y_pred = lstm_model.predict([photo, sequence], verbose=0)
        y_pred = np.argmax(y_pred[0])
        word = indices_to_words[y_pred]
        in_text += " " + word
        if word == "<end>":
            break

    # Remove "<start>" and "<end>" tokens, and filter out "Unk"
    final = [word for word in in_text.split()[1:-1] if word != "Unk"]
    print("Greedy search complete.")
    return final

def beam_search(photo, k):
    """Generate a caption using beam search."""
    print("Starting beam search...")
    photo = photo.reshape(1, 2048)
    in_text = "<start>"
    sequence = [words_to_indices[s] for s in in_text.split() if s in words_to_indices]
    sequence = pad_sequences([sequence], maxlen=max_length, padding="post")
    y_pred = lstm_model.predict([photo, sequence], verbose=0)
    predicted = []
    y_pred = y_pred.reshape(-1)
    for i in range(y_pred.shape[0]):
        predicted.append((i, y_pred[i]))
    predicted = sorted(predicted, key=lambda x: x[1])[::-1]
    b_search = []
    for i in range(k):
        word = indices_to_words[predicted[i][0]]
        b_search.append((in_text + " " + word, predicted[i][1]))

    for idx in range(max_length):
        b_search_square = []
        for text in b_search:
            if text[0].split(" ")[-1] == "<end>":
                break
            sequence = [words_to_indices[s] for s in text[0].split() if s in words_to_indices]
            sequence = pad_sequences([sequence], maxlen=max_length, padding="post")
            y_pred = lstm_model.predict([photo, sequence], verbose=0)
            predicted = []
            y_pred = y_pred.reshape(-1)
            for i in range(y_pred.shape[0]):
                predicted.append((i, y_pred[i]))
            predicted = sorted(predicted, key=lambda x: x[1])[::-1]
            for i in range(k):
                word = indices_to_words[predicted[i][0]]
                b_search_square.append((text[0] + " " + word, predicted[i][1] * text[1]))

        if len(b_search_square) > 0:
            b_search = (sorted(b_search_square, key=lambda x: x[1])[::-1])[:k]

    # Remove "<start>" and "<end>" tokens, and filter out "Unk"
    final = [word for word in b_search[0][0].split()[1:-1] if word != "Unk"]
    print("Beam search complete.")
    return final

# Extracting image encodings (features) from ResNet50 and forming dict test_features
path = "all_images/Flicker8k_Dataset/"
folder_path_images = r"C:\Work\BITS\ImageCoaptioning\Flicker8k_Dataset"
test_features = {}

# Instructions to run the application
if __name__ == "__main__":
    import uvicorn

    print("Starting FastAPI server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
