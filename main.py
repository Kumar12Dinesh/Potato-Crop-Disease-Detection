import streamlit as st
import tensorflow as tf
import numpy as np


#--> Tensorflow Model Prediction
def model_prediction(test_image):
  model=tf.keras.models.load_model('trained_model3.keras')# most importnat phase 
  image= tf.keras.preprocessing.image.load_img(test_image,target_size=(256,256))
  input_arr=tf.keras.preprocessing.image.img_to_array(image)
  input_arr=np.array([input_arr]) #--> Convert Single image into batch
  prediction=model.predict(input_arr)
  result_index=np.argmax(prediction)
  return result_index



#--> Sidebar
st.sidebar.title("Dashboard")
app_mode= st.sidebar.selectbox("Select Page",["Home","About","Disease Prediction"]) 


#--> Defining Of Home Page

if(app_mode=="Home"):
  st.header("CROP DISEASE RECOGNITION SYSTEM")
  image_path=r"C:\Users\navu1\OneDrive\Desktop\fukayamamo-kj8lssgzULM-unsplash.jpg"
  st.image(image_path,use_column_width=True)
  st.markdown("""
              Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!
    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.
    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.
    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    
              """)
  
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 1500 rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (900 Images)
                2. test (300 Images)
                3. validation (300 Images)
                """)
    

elif(app_mode=="Disease Prediction"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = [
            "Potato/Test/Potato___Early_blight",
             "Potato/Test/Potato___healthy",
             "Potato/Test/Potato___Late_blight"
            ]
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))