### Health Management app
from dotenv import load_dotenv

load_dotenv() ##load all env files

import streamlit as st

import os
from PIL import Image
import google.generativeai as genai

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

### Function to load Google Pro Vision API and get response

def get_gemini_response(input,image,prompt):
    model=genai.GenerativeModel("gemini-pro-vision")
    response=model.generate_content([input,image[0],prompt])
    return response.text

def input_image_setup(uploaded_file):
    #check if a file has been uploaded
    if uploaded_file is not None:
        #read the file into bytes
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                'mime_type': uploaded_file.type, #get the mime type of the uploaded file
                'data':bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError('no file uploaded')

##initializ the stremilit app

st.set_page_config(page_title='Gemini Health App')

st.header('Gemini Health App')
input = st.text_input('Input Prompt:', key='input')
uploaded_file = st.file_uploader('Choose an image of the food...',type=['jpg','jpeg','png'])
image = ''
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image,caption='uploaded Image',use_column_width=True) 


submit = st.button('Tell me the total calories')

input_prompt = '''
you are an expert in nutritionist where you need to see the food items frm the image
and caluclate the total calories,also provide the details of every food item with calories intake
in below format
1. Item 1 - no of calories
2. Item 2 - no of calories
-----
-----

'''
##  if submit button is clicked

if submit :
    image_data = input_image_setup(uploaded_file)
    response = get_gemini_response(input_prompt,image_data,input)
    st.subheader('The Response')
    st.write(response)
