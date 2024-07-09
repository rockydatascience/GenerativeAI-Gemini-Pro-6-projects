import streamlit as st
from PIL import Image
import os
import google.generativeai as genai
import textwrap

# Configure the Generative AI API with the provided API key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Function to convert text to Markdown format
def to_markdown(text):
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

# Function to get the Gemini AI response
def get_gemini_response(input_text, image):
    model = genai.GenerativeModel('gemini-pro-vision')
    if input_text:
        response = model.generate_content([input_text, image])
    else:
        response = model.generate_content(image)
    return response.text

# Initialize Streamlit app
st.set_page_config(page_title='IMAGE Classifiation')
st.header('GEMINI AI IMAGE APP ANALYSIS')

# Input prompt from the user
input_text = st.text_input('Input prompt:', key='input')
upload_file = st.file_uploader('Choose an image', type=['jpg', 'png', 'jpeg'])

# Display the uploaded image
image = None
if upload_file is not None:
    image = Image.open(upload_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

# Button to submit and get response
submit = st.button('Explain brief about Image')

# When submit button is clicked
if submit:
    if image is not None:
        response = get_gemini_response(input_text, image)
        st.subheader('The response is')
        st.markdown(to_markdown(response))  # Directly pass indented text to st.markdown
    else:
        st.warning('Please upload an image.')
