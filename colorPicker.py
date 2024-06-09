'''
    Nama  : Muhammad Faiz Fahri
    NPM   : 140810220002
    Kelas : B
'''

import streamlit as st
import cv2
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

def load_image(image_file):
    img = Image.open(image_file)
    return img

def find_dominant_colors(img, k=8):
    img = np.array(img)
    if img.shape[2] == 4:  
        img = img[:, :, :3] 
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(img)
    colors = kmeans.cluster_centers_
    colors = colors.round(0).astype(int)
    colors = sorted(colors, key=lambda x: np.sqrt(0.299*x[0]**2 + 0.587*x[1]**2 + 0.114*x[2]**2))
    return colors

st.set_page_config(page_title="Color Picker dari Image", page_icon="🎨")
st.title('Color Picker dari Image 🎨')
st.caption('Muhammad Faiz Fahri 140810220002')

default_image_path = "example.jpg" 
default_image = load_image(default_image_path)

image_file = st.file_uploader("Upload Image", type=['png', 'jpeg', 'jpg'])

num_colors = st.slider('Pilih berapa banyak warna dominan', 3, 10, 8)

if image_file is not None:
    image = load_image(image_file)
    st.image(image, caption='Gambar yang diupload', use_column_width=True)
    dominant_colors = find_dominant_colors(image, k=num_colors)
else:
    st.image(default_image, caption='Contoh Gambar', use_column_width=True)
    dominant_colors = find_dominant_colors(default_image, k=num_colors)

st.subheader('Warna Dominan:')
palette_html = '<div style="display: flex; border: 2px solid white; width: fit-content;">'
for color in dominant_colors:
    color_code = f'rgb({color[0]}, {color[1]}, {color[2]})'
    palette_html += f'<div style="width:75px; height:75px; background-color:{color_code};"></div>'
palette_html += '</div>'

st.markdown(palette_html, unsafe_allow_html=True)