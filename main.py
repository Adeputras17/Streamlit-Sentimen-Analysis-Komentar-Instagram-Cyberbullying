import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import joblib
import csv
from transformers import pipeline
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from tensorflow.keras.models import load_model

st.title('komentar Instagram Cyberbullyng')
st.markdown('Aplikasi ini tentang sentimen analisis instagram mengenai cyberbullyng')

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.image import load_img,img_to_array

from tensorflow.python.keras import utils


model = tf.keras.models.load_model("streamlit/instagram", 'rb')



form = st.form(key='sentiment-form')
user_input = form.text_area('ketik di sini')
submit = form.form_submit_button('Submit')
prediction = data.predict(input_variables)   
result = prediction(user_input)[0]    
label = result['label']    
score = result['score']