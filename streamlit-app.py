# Sebastian Pedreros - Felipe Serey - Jose Poblete

import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


model = load_model('models/mini_alexnet_rps.h5')
class_names = ['paper', 'rock', 'scissors']

st.title("Clasificador Rock-Paper-Scissors con Mini-AlexNet")

uploaded_file = st.file_uploader("Sube una imagen de rock, paper o scissors", type=['png','jpg','jpeg'])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Imagen cargada', use_column_width=True)
    img = image.resize((150,150))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    st.write(f"Predicción: **{class_names[idx]}**")
    st.write(f"Confianza: **{preds[idx]*100:.2f}%**")
