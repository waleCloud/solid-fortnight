import numpy as np
import streamlit as st
import tensorflow
import cv2 as cv
from keras.models import load_model

sign_labels = {
  0: 'Speed limit (20km/h)',
  1: 'Speed limit (30km/h)',
  2: 'Speed limit (50km/h)',
  3: 'Speed limit (60km/h)',
  4: 'Speed limit (70km/h)',
  5: 'Speed limit (80km/h)',
  6: 'End of speed limit (80km/h)',
  7: 'Speed limit (100km/h)',
  8: 'Speed limit (120km/h)',
  9: 'No passing',
  10: 'No passing vehicle over 3.5 tons',
  11: 'Right-of-way at intersection',
  12: 'Priority road',
  13: 'Yield',
  14: 'Stop',
  15: 'No vehicles',
  16: 'Vehicle > 3.5 tons prohibited',
  17: 'No entry',
  18: 'General caution',
  19: 'Dangerous curve left',
  20: 'Dangerous curve right',
  21: 'Double curve',
  22: 'Bumpy road',
  23: 'Slippery road',
  24: 'Road narrows on the right',
  25: 'Road work',
  26: 'Traffic signals',
  27: 'Pedestrians',
  28: 'Children crossing',
  29: 'Bicycles crossing',
  30: 'Beware of ice/snow',
  31: 'animals crossing',
  32: 'End speed + passing limits',
  33: 'Turn right ahead',
  34: 'Turn left ahead',
  35: 'Ahead only',
  36: 'Go straight or right',
  37: 'Go straight or left',
  38: 'Keep right',
  39: 'Keep left',
  40: 'Roundabout mandatory',
  41: 'End of no passing',
  42: 'End no passing vehicle > 3.5 tons'
}

def load_image(img, show=False):

  img_tensor = tensorflow.keras.preprocessing.image.img_to_array(img)
  img_tensor = np.expand_dims(img_tensor, axis= 0)
  img_tensor /= 255.

  return img_tensor

@st.cache
def sign_predict(image):
  # load model
  model = load_model('cnn_model.h5', compile=False)

  prediction = model.predict(image)
  prediction_max = np.argmax(prediction)
  prediction_label = sign_labels[prediction_max]
  confidence = np.max(prediction)

  print('confidence', confidence)
  print('prediction_label', prediction_label)
  return prediction_label, confidence


def main():
  st.title('Traffic Signs Classifier')
  st.markdown("""
      This application classifies traffic signs. Upload any photo of a traffic sign 
      and receive its name out of 43 present classes. For getting the correct prediction, 
      try to upload a square picture containing only the sign.
      """)
  with st.expander("See list of classes"):
    st.write(list(sign_labels.values()))
  
  uploaded_image = st.file_uploader('Upload a photo of traffic sign here', type=['jpg', 'jpeg', 'png'])

  if uploaded_image is not None:
    col1, col2 = st.columns(2)
    col1.markdown('#### Your picture ####')
    # image = Image.open(uploaded_image)
    with col1:
      # Convert the file to an opencv image.
      file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
      opencv_image = cv.imdecode(file_bytes, 1)
      resized_image = cv.resize(opencv_image, (50, 50))
      
      # Now do something with the image! For example, let's display it:
      st.image(resized_image, use_column_width=True, channels="BGR")

    procesed_image = load_image(resized_image)

    # Make prediction
    prediction_label, confidence = sign_predict(procesed_image)
    st.write('#### Prediction label----: ', prediction_label)
    st.write('### Confidence: ', confidence)

    st.markdown('***')



if __name__ == '__main__':
  main()