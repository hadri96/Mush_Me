import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# image
image = Image.open('logo.jpg')
st.image(image, width=100, use_column_width=None)

# title
st.markdown("""# MUSH ME
## You went mushroom picking and you wonder if you can eat a mushroom?
Upload your picture and check it by yourself! """)

# file uploader
uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=True)

# image
image = Image.open('mushroom.jpg')
st.image(image)


