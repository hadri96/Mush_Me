import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

#SIDE BAR
logo = Image.open('front_end/mush_me_logo.jpg')
st.sidebar.image(logo, width=200, use_column_width=None)
st.sidebar.title('ABOUT')
st.sidebar.write("The aim of this project is to help fervent mushroom pickers to avoid intoxication. This model has been trained using the Danish Fungi Dataset.")
st.sidebar.write("This application was developped by four students in Data Science @LeWagon #batch619.")
#title
st.title("THE MUSH ME PROJECT")

st.write('You went mushroom picking and you wonder if you can eat a mushroom? This app helps you verify it by yourself!')

image = Image.open('front_end/beautiful_mushroom.jpeg')
st.image(image, width=None, use_column_width=None)

st.header("DRAG & DROP")
                
# file uploader
uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=True)

#submit button
submit = st.button("Submit")

if submit:
    st.write("You succesfully uploaded your mushroom picture.")
else :
    st.write("Sorry, something went wrong with your upload")    
st.header("LET THE MAGIC HAPPEN")

st.header("CAN YOU EAT IT?")


#output display data
#st.dataframe(data=None, width=None, height=None)

#output display media
#st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')