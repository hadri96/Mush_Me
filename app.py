import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

#title
st.title("DISCOVER MUSH ME")

# image
image = Image.open('front_end/mush_me_logo.jpg')
col1, col2, col3 = st.beta_columns([1,6,1])

with col1:
    st.image(image, width=200, use_column_width=None)

with col2:
    st.write("")

with col3:
    st.write("")

st.write("You went mushroom picking and you wonder if you can eat a mushroom? Upload your picture and check it by yourself!")

#SIDE BAR

st.sidebar.write("UPLOAD YOUR MUSHROOM PICTURE")
                 
# file uploader
uploaded_file = st.sidebar.file_uploader("Choose a file", accept_multiple_files=True)

#submit button
submit = st.sidebar.button("Submit")

if submit:
    st.sidebar.write("You succesfully uploaded your file!")

#layout left
st.sidebar.uploaded_file

#output display data
#st.dataframe(data=None, width=None, height=None)

#output display media
#st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')