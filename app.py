import streamlit as st
from PIL import Image
from googletrans import Translator
from gtts import gTTS
import os

def translate_text(text, target_language):
    translator = Translator()
    translated_text = translator.translate(text, dest=target_language)
    return translated_text.text


def text_to_speech(text, language="mr"):
    tts = gTTS(text=text, lang=language, slow=False)
    # Save the audio to a file
    tts.save("output.mp3")

st.title("Blind Assistance System")
st.write("---")
# Take inputs from the streamlit
model_name = st.sidebar.selectbox("Select Pre-trained Model:", ("VIT-GPT-2", "BLIP-Large", "BLIP-Base"))
st.write("The purpose of this project is to build a blind assistance system using custom or existing pre-trained models. I have implemented it using 3 different models.")
st.write("1]VIT-GPT2: The Vision Encoder Decoder Model has been used to initialize an image-to-text model using VIT as the encoder and the pre-trained language model GPT2 as the decoder.")
st.write("2]BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation: a)Large and b) Base. BLIP is a new Vision Language Processing framework which transfers flexibly to both vision-language understanding and generation tasks. BLIP effectively utilizes the noisy web data by bootstrapping the captions, where a captioner generates synthetic captions and a filter removes the noisy ones.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

from transformers import pipeline

if uploaded_file is not None:

 if model_name == "VIT-GPT-2":
    st.write("VIT-GPT-2 model is selected.")
    image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write(image_to_text(image)[0]["generated_text"])
    text_to_speech(image_to_text(image)[0]["generated_text"])
    st.audio("output.mp3")
 elif model_name == "BLIP-Large":
    st.write("BLIP-Large model is selected.")
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write(image_to_text(image)[0]["generated_text"])
    text_to_speech(image_to_text(image)[0]["generated_text"])
    st.audio("output.mp3")
 else:
    st.write("BLIP-Base model is selected.")
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write(image_to_text(image)[0]["generated_text"])
    text_to_speech(image_to_text(image)[0]["generated_text"])
    st.audio("output.mp3")
