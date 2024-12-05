import requests
import numpy as np
import json
import cv2

url = 'https://tensorflow-faces-model.onrender.com/v1/models/faces-model:predict'

def prepare_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128)) 
    img = img / 255.0 
    img = np.expand_dims(img, axis=0)  
    return img.tolist() 

image_path = 'Enigma.jpg'  

input_data = prepare_image(image_path)

data = {
    "signature_name": "serving_default",  
    "instances": input_data
}

response = requests.post(url, json=data)

if response.status_code == 200:
    predictions = response.json()
    print("Predicciones:", predictions)
else:
    print("Error en la petición:", response.status_code, response.text)
