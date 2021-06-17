import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

#SIDE BAR
logo = Image.open('front_end/mush_me_logo.jpg')
st.sidebar.image(logo, width=280, use_column_width=None)
st.sidebar.title('ABOUT')
st.sidebar.write("The aim of this project is to help fervent mushroom pickers to avoid intoxication. This model has been trained using the Danish Fungi Dataset.")
st.sidebar.write("This application was developped by four students in Data Science @LeWagon #batch619.")

st.sidebar.header('**DISCLAIMER**')
st.sidebar.write('_The results provided by this application are predictions and should be always be cross-validated by a mushroom expert before eating._')

#TITLE
st.title("THE MUSH ME PROJECT")

#1st STEP
st.write('**You went mushroom picking and you wonder if you can eat a mushroom? This app helps you verify it by yourself!**')

image = Image.open('front_end/beautiful_mushroom.jpeg')
st.image(image, width=None, use_column_width=None)

st.header("DRAG & DROP")         
uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=True) 

monthes = ['January', 'February','March','April','May','June','July','August','September','October','November','December']
st.selectbox('SELECT A MONTH:', monthes, index=0, format_func=str, key=None, help=None)   

substrate = ['Soil', 'Dead wood']
st.selectbox('SELECT A SUBSTRATE:', substrate, index=0, format_func=str, key=None, help=None)   
    
habitat = ['Forest','Mountains']
st.selectbox('SELECT AN ENVIRONMENT:', habitat, index=0, format_func=str, key=None, help=None)   

submit = st.button("Submit")
if submit:
    st.write("You succesfully uploaded your mushroom picture.")
else :
    st.write("Sorry, something went wrong with your upload")   

#2ND STEP
st.header("LET THE MAGIC HAPPEN")

st.progress(2)


col1, col2 = st.beta_columns(2)
with col1:
    st.subheader('YOUR PICTURE')
    st.image(uploaded_file, width=None, use_column_width=None)  
with col2:
    st.subheader('OUR GUESS')
    st.image(uploaded_file, width=None, use_column_width=None)

#output display data

st.write('RESULTS')
st.write(pd.DataFrame({
'Mushroom Name': ['Morille', 'Chanterelle'],
'Probability': ['80%', '20%'],
'Edibility': ['True', 'False'],
'Comment': ['Careful, morel season is between september and october', 'No comment'],
}))

#st.dataframe(data=None, width=None, height=None)


#3RD STEP
st.header("CAN YOU EAT IT?")


