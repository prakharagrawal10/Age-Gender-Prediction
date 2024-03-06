import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
from tqdm.notebook import tqdm
warnings.filterwarnings("ignore")

import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten,  MaxPooling2D,  Input


BASE_DIR = r'C:\Users\prakh\Desktop\archive\UTKFace'
#labels = age, gender, ethnicity
image_paths= []
age_labels= []
gender_labels=[]

for filename in tqdm(os.listdir(BASE_DIR)):
    image_path = os.path.join(BASE_DIR, filename)
    temp = filename.split('_')
    age=int(temp[0])
    gender=int(temp[1])
    image_paths.append(image_path)
    age_labels.append(age)
    gender_labels.append(gender)
    
df=pd.DataFrame()
df['image'],  df['age'], df['gender'] = image_paths, age_labels, gender_labels
print(df)

gender_dict={0:'Male',1:'Female'}

from PIL import Image

def extract_features(images):
    features=[]
    for image in tqdm(images):
        img=load_img(image)
        img=img.resize((128,128),Image.Resampling.LANCZOS)
        img=np.array(img)
        features.append(img)
        
    features=np.array(features)
    return features       
        
x=extract_features(df['image'])

x=x/255.0
y_gender=np.array(df['gender'])
y_age=np.array(df['age'])
input_shape=(128,128,3)



import cv2
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model

# Load your pre-trained model
model = load_model('age_gender_model(final).h5')

# Open a connection to the camera (0 represents the default camera)
cap = cv2.VideoCapture(0)

def preprocess_camera_frame(frame):
    # Resize frame to 128x128
    frame = cv2.resize(frame, (128, 128))

    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Normalize pixel values to [0, 1]
    frame = frame / 255.0

    return frame

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Perform preprocessing
    preprocessed_frame = preprocess_camera_frame(frame)

    # Expand dimensions to match the model's expected input shape
    input_data = np.expand_dims(preprocessed_frame, axis=0)

    # Perform age and gender prediction
    predicted_gender, predicted_age = model.predict(input_data)

    # Convert gender prediction to 'Male' or 'Female'
    gender_prediction = "Male" if predicted_gender[0] < 0.5 else "Female"

    # Display predictions on the frame
    cv2.putText(frame, f"Gender: {gender_prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Age: {int(predicted_age[0])} years", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Camera Input', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
