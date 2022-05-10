import streamlit as st
from PIL import Image
from helper import return_closest_images


st.title("ADBI Capstone - Distributed Image Similarity Detection with Deep Learning and PySpark")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:

    print(uploaded_file)
    image = Image.open(uploaded_file)
    print(type(image))
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    BASE_PATH = "cars_data/cars_train/cars_train/"
    file_path = BASE_PATH + uploaded_file.name

    print(file_path)

    top5 = return_closest_images(file_path)

    st.title("Top 5 similar images")
    st.image(BASE_PATH + top5[0], caption='Image 1', use_column_width=True)
    st.image(BASE_PATH + top5[1], caption='Image 2', use_column_width=True)
    st.image(BASE_PATH + top5[2], caption='Image 3', use_column_width=True)
    st.image(BASE_PATH + top5[3], caption='Image 4', use_column_width=True)
    st.image(BASE_PATH + top5[4], caption='Image 5', use_column_width=True)
