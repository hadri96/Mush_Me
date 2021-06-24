import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

#SIDE BAR
logo = Image.open('new-logo.jpeg')
st.sidebar.image(logo, width=280, use_column_width=None)
st.sidebar.title('ABOUT')
st.sidebar.write("The aim of this project is to help fervent mushroom pickers to avoid intoxication. This model has been trained using the Danish Fungi Dataset.")
st.sidebar.write("This application was developped by four students in Data Science @LeWagon #batch619.")

st.sidebar.header('**DISCLAIMER**')
st.sidebar.write('_The results provided by this application are predictions and should be always be cross-validated by a mushroom expert before eating._')

#TITLE
st.title("MUSH ME")

#1st STEP
st.write('You went mushroom picking and you wonder if you can eat it? Verify it by yourself!')

st.header("DRAG & DROP YOUR PICTURE")         
uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=True) 

submit = st.button("Predict")
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
'Mushroom Name': ['MODEL INPUT'],
'Probability': ['MODEL INPUT'],
'Edibility': ['MODEL INPUT'],
'Comment': ['Careful, morel season is between september and october'],
}))

# Comments
# IF MONTH=FALSE: Careful! The harvest season of this mushroom does not match with the month you have selected.
# IF SUBSTRATE=FALSE: Careful! The substrate of this mushroom does not match with the substrate you have selected.
# IF ENVIRONMENT=FALSE: Careful! The environment of this mushroom does not match with the environment you have selected.
# ELSE: No comment

#st.dataframe(data=None, width=None, height=None)



