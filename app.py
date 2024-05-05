import streamlit as st
from PIL import Image

st.title("Blind Navigation System")
st.write("---")

# Take inputs from the streamlit
model_name = st.sidebar.selectbox("Select Custom Model for Object Detection:", ("VIT-GPT-2", "BLIP-Large", "BLIP-Base"))

st.write("The purpose of this project is to design a blind navigation system using existing custom pre-trained models. I have implemented three different models for the same.")
st.write("1.] BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation- a)Base and b)Large trained on COCO dataset.2] VIT-GPT-2 based custom model")

from transformers import pipeline

image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
from googletrans import Translator
from gtts import gTTS
import os

def translate_text(text, target_language):
    translator = Translator()
    translated_text = translator.translate(text, dest=target_language)
    return translated_text.text

def text_to_speech(text, language='mr'):
    tts = gTTS(text=text, lang=language, slow=False)
    # Save the audio to a file
    tts.save("output.mp3")


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write(image_to_text(image)[0]['generated_text'])
    text_to_speech(image_to_text(image)[0]['generated_text'])
    st.audio("output.mp3")
    if model_name == "Vit-GPT-2":
        st.write("Vit-GPT-2 model is selected.")

    # st.write("")
    # st.write("Generating caption...")
    # caption = generate_caption(image)
    # st.write(caption)