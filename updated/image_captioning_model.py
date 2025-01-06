# Standard library imports
import os
import pickle

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
# Third-party library imports
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Add, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


class ImageCaptioning:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.train_captions = {}
        self.test_captions = {}
        self.validation_captions = {}
        self.train_features = {}
        self.test_features = {}
        self.validation_features = {}
        self.words_to_indices = {}
        self.indices_to_words = {}
        self.vocab_size = 0
        self.max_length = 40

        self.model = self.build_model()

    def load_data(self):
        """Load and preprocess data."""
        print(os.getcwd())
        image_tokens = pd.read_csv(f"{self.folder_path}\\Flickr8k.lemma.token.txt", sep='\t',
                                   names=["img_id", "img_caption"])
        train_image_names = pd.read_csv(f"{self.folder_path}\\Flickr_8k.trainImages.txt", names=["img_id"])
        test_image_names = pd.read_csv(f"{self.folder_path}\\Flickr_8k.testImages.txt", names=["img_id"])
        val_image_names = pd.read_csv(f"{self.folder_path}\\Flickr_8k.devImages.txt", names=["img_id"])

        # Preprocess image tokens
        image_tokens["img_id"] = image_tokens["img_id"].map(lambda x: x[:len(x) - 2])
        image_tokens["img_caption"] = image_tokens["img_caption"].map(lambda x: "<start> " + x.strip() + " <end>")

        # Create dictionaries for train, test, and validation captions
        self.train_captions = self.create_captions_dict(train_image_names, image_tokens)
        self.test_captions = self.create_captions_dict(test_image_names, image_tokens)
        self.validation_captions = self.create_captions_dict(val_image_names, image_tokens)

    def create_captions_dict(self, image_names, image_tokens):
        """Create a dictionary mapping image ID to captions."""
        captions = {}
        for i in tqdm(range(len(image_names))):
            img_id = image_names["img_id"].iloc[i]
            captions[img_id] = [caption for caption in image_tokens[image_tokens["img_id"] == img_id].img_caption]
        return captions

    def build_model(self):
        """Build and compile the image captioning model."""
        model = ResNet50(include_top=False, weights='imagenet', pooling='avg', input_shape=(224, 224, 3))
        model.summary()
        return model

    def encode_images(self, image_names, features_dict):
        """Extract image encodings from the model and save them."""
        folder_path_images = r"C:\Work\BITS\ImageCoaptioning\Flicker8k_Dataset"
        for image_name in tqdm(image_names):
            img_path = f"{folder_path_images}\\{image_name}"
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = self.model.predict(x)
            features_dict[image_name] = features.squeeze()

    def save_encoded_images(self):
        """Save encoded images for train, test, and validation sets."""
        self.encode_images(self.train_captions.keys(), self.train_features)
        with open("train_encoded_images.p", "wb") as pickle_f:
            pickle.dump(self.train_features, pickle_f)

        self.encode_images(self.test_captions.keys(), self.test_features)
        with open("test_encoded_images.p", "wb") as pickle_f:
            pickle.dump(self.test_features, pickle_f)

        self.encode_images(self.validation_captions.keys(), self.validation_features)
        with open("validation_encoded_images.p", "wb") as pickle_f:
            pickle.dump(self.validation_features, pickle_f)

    def prepare_vocabulary(self):
        """Prepare vocabulary and mappings for captions."""
        all_captions = [caption for img_id in tqdm(self.train_captions) for caption in self.train_captions[img_id]]
        unique_words = list(set(" ".join(all_captions).strip().split(" ")))
        self.vocab_size = len(unique_words) + 1

        self.words_to_indices = {val: index + 1 for index, val in enumerate(unique_words)}
        self.indices_to_words = {index + 1: val for index, val in enumerate(unique_words)}
        self.words_to_indices["Unk"] = 0
        self.indices_to_words[0] = "Unk"

        # Save mappings
        joblib.dump(self.words_to_indices, "words_to_indices.pkl")
        joblib.dump(self.indices_to_words, "indices_to_words.pkl")

    def encode_captions(self):
        """Encode captions into sequences."""
        train_encoded_captions = {}
        for img_id in tqdm(self.train_captions):
            train_encoded_captions[img_id] = []
            for caption in self.train_captions[img_id]:
                train_encoded_captions[img_id].append([self.words_to_indices[word] for word in caption.split(" ")])

        # Pad sequences
        for img_id in tqdm(train_encoded_captions):
            train_encoded_captions[img_id] = pad_sequences(train_encoded_captions[img_id], maxlen=self.max_length,
                                                           padding='post')

        return train_encoded_captions

    def data_generator(self, train_encoded_captions, num_of_photos):
        """Generate data for training."""
        X1, X2, Y = list(), list(), list()
        n = 0
        for img_id in train_encoded_captions:
            n += 1
            for i in range(5):  # Loop over 5 captions per image
                for j in range(1, 40):  # Loop to create sequences
                    curr_sequence = train_encoded_captions[img_id][i][0:j].tolist()
                    next_word = train_encoded_captions[img_id][i][j]
                    curr_sequence = pad_sequences([curr_sequence], maxlen=self.max_length, padding='post')[0]
                    one_hot_next_word = to_categorical([next_word], self.vocab_size)[0]
                    X1.append(self.train_features[img_id])  # Image features
                    X2.append(curr_sequence)  # Input sequence
                    Y.append(one_hot_next_word)  # Target word (one-hot encoded)
            if n == num_of_photos:
                yield (np.array(X1), np.array(X2)), np.array(Y)
                X1, X2, Y = list(), list(), list()  # Reset for the next batch
                n = 0

    def build_captioning_model(self):
        """Build the captioning model architecture."""
        input_1 = Input(shape=(2048,))
        dropout_1 = Dropout(0.2)(input_1)
        dense_1 = Dense(256, activation='relu')(dropout_1)

        input_2 = Input(shape=(self.max_length,))
        embedding_1 = Embedding(self.vocab_size, 256)(input_2)
        dropout_2 = Dropout(0.2)(embedding_1)
        lstm_1 = LSTM(256)(dropout_2)

        add_1 = Add()([dense_1, lstm_1])  # Functional API call
        dense_2 = Dense(256, activation='relu')(add_1)
        dense_3 = Dense(self.vocab_size, activation='softmax')(dense_2)

        model = Model(inputs=[input_1, input_2], outputs=dense_3)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train_model(self, train_encoded_captions, epochs=5, no_of_photos=5):
        """Train the captioning model."""
        steps = len(train_encoded_captions) // no_of_photos
        for i in range(epochs):
            generator = self.data_generator(train_encoded_captions, no_of_photos)
            self.model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

    def save_model(self):
        """Save the trained model."""
        joblib.dump(self.model, "image_captioning_model.joblib")
        print("Model saved as image_captioning_model.joblib")

    def greedy_search(self, photo):
        """Greedy search for generating captions."""
        photo = photo.reshape(1, 2048)
        in_text = '<start>'
        for i in range(self.max_length):
            sequence = [self.words_to_indices[s] for s in in_text.split(" ") if s in self.words_to_indices]
            sequence = pad_sequences([sequence], maxlen=self.max_length, padding='post')
            y_pred = self.model.predict([photo, sequence], verbose=0)
            y_pred = np.argmax(y_pred[0])
            word = self.indices_to_words[y_pred]
            in_text += ' ' + word
            if word == '<end>':
                break
        return in_text.split()[1:-1]

    def beam_search(self, photo, k):
        """Beam search for generating captions."""
        photo = photo.reshape(1, 2048)
        in_text = '<start>'
        sequence = [self.words_to_indices[s] for s in in_text.split(" ") if s in self.words_to_indices]
        sequence = pad_sequences([sequence], maxlen=self.max_length, padding='post')
        y_pred = self.model.predict([photo, sequence], verbose=0)
        predicted = [(i, score) for i, score in enumerate(y_pred.flatten())]
        predicted = sorted(predicted, key=lambda x: x[1], reverse=True)

        b_search = [(in_text + ' ' + self.indices_to_words[predicted[i][0]], predicted[i][1]) for i in range(k)]
        for idx in range(self.max_length):
            b_search_square = []
            for text in b_search:
                if text[0].split(" ")[-1] == "<end>":
                    break
                sequence = [self.words_to_indices[s] for s in text[0].split(" ") if s in self.words_to_indices]
                sequence = pad_sequences([sequence], maxlen=self.max_length, padding='post')
                y_pred = self.model.predict([photo, sequence], verbose=0)
                predicted = [(i, score) for i, score in enumerate(y_pred.flatten())]
                predicted = sorted(predicted, key=lambda x: x[1], reverse=True)
                for i in range(k):
                    word = self.indices_to_words[predicted[i][0]]
                    b_search_square.append((text[0] + ' ' + word, predicted[i][1] * text[1]))
            if len(b_search_square) > 0:
                b_search = sorted(b_search_square, key=lambda x: x[1], reverse=True)[:k]
        final = b_search[0][0].split()[1:-1]
        return final

    def predict_using_beam_search(self, k=3):
        """Predict captions using beam search."""
        for img_id in self.test_features:
            img = cv2.imread(os.path.join(self.folder_path, img_id))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()
            photo = self.test_features[img_id]
            candidate = self.beam_search(photo, k)
            print("Predicted Caption: ", " ".join(candidate))

    def calculate_bleu_score(self):
        """Calculate average BLEU score on the test set."""
        total_score = 0
        for img_id in tqdm(self.test_features):
            photo = self.test_features[img_id]
            reference = [caps.split(" ")[1:-1] for caps in self.test_captions[img_id]]
            candidate = self.greedy_search(photo)
            score = sentence_bleu(reference, candidate)
            total_score += score
        avg_score = total_score / len(self.test_features)
        print("Avg BLEU Score: ", avg_score)


# Usage
folder_path = r"C:\Work\BITS\ImageCoaptioning\Flickr8k_text"
image_captioning = ImageCaptioning(folder_path)
image_captioning.load_data()
image_captioning.save_encoded_images()
image_captioning.prepare_vocabulary()
train_encoded_captions = image_captioning.encode_captions()
image_captioning.train_model(train_encoded_captions)
image_captioning.save_model()
image_captioning.predict_using_beam_search()
image_captioning.calculate_bleu_score()