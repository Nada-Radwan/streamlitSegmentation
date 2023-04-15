import streamlit as st

import tensorflow as tf
from PIL import Image
import numpy as np
import cv2


from huggingface_hub import from_pretrained_keras
try:
    model=from_pretrained_keras("Nada2001/streamlitSeg")
except:
    model=tf.keras.models.load_model('/content/drive/MyDrive/dataX-ray_modelH5/UNet data xray_model3.h5')
    pass
    
st.header("Segmentation of Teeth in Panoramic X-ray Image Using UNet")

examples=["1.jpg","2.jpg","3.jpg"]

def load_image(image_file):
	img = Image.open(image_file)
	return img

def convert_one_channel(img):
    #some images have 3 channels , although they are grayscale image
    if len(img.shape)>2:
        img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    else:
        return img
    
def convert_rgb(img):
    #some images have 3 channels , although they are grayscale image
    if len(img.shape)==2:
        img= cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)  
        return img
    else:
        return img
    
    
st.subheader("Upload Dental Panoramic X-ray Image Image")
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])


col1, col2, col3 = st.columns(3)
with col1:
    ex=load_image(examples[0])
    st.image(ex,width=200)
    if st.button('Example 1'):
        image_file=examples[0]

with col2:
    ex1=load_image(examples[1])
    st.image(ex1,width=200)
    if st.button('Example 2'):
        image_file=examples[1]


with col3:
    ex2=load_image(examples[2])
    st.image(ex2,width=200)
    if st.button('Example 3'):
        image_file=examples[2]
    
    
if image_file is not None:

      img=load_image(image_file)
      
      st.text("Making A Prediction ....")
      st.image(img,width=850)
      
      img=np.asarray(img)
  
      img_cv=convert_one_channel(img)
      img_cv=cv2.resize(img_cv,(512,512), interpolation=cv2.INTER_LANCZOS4)
      img_cv=np.float32(img_cv/255)
      
      img_cv=np.reshape(img_cv,(1,512,512,1))
      prediction=model.predict(img_cv)
      predicted=prediction[0]
      predicted = cv2.resize(predicted, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
      mask=np.uint8(predicted*255)# 
      _, mask = cv2.threshold(mask, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
      kernel =( np.ones((5,5), dtype=np.float32))
      mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations=1 )  
      mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations=1 )
      cnts,hieararch=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
      output = cv2.drawContours(convert_rgb(img), cnts, -1, (255, 0, 0) , 3)


      if output is not None :      
          st.subheader("Predicted Image")  
          st.write(output.shape)
          st.image(output,width=850)

      st.text("DONE ! ....")
