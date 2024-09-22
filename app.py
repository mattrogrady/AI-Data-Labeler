import streamlit as st
from PIL import Image
import numpy as np
import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

st.title('Stop Sign Detection App')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
	image=Image.open(uploaded_file)
	st.image(image, caption='Uploaded Image', use_column_width=True)

if uploaded_file is not None:
	img_array = np.array(image)
	
	results = model(img_array)

	detected_objects = results.pandas().xyxy[0]
	stop_signs = detected_objects[detected_objects['name'] == 'stop sign']

	for index, row in stop_signs.iterrows():
		x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
		cv2.rectangle(img_array, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
	
	img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
	st.image(img_array, caption='Processed Image', use_column_width=True)
