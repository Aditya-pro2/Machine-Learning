import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import tensorflow as tf

st.title(":blue[HANDWRITTEN DIGIT RECOGNIZER]")

def load(p):
    try:
        with tf.keras.backend.name_scope('model_loading'):
            model = tf.keras.models.load_model(p, compile = False)
        return model
    except IndexError as e:
        print("IndexError:", e)
        return None
    except Exception as e:
        print("Error loading the model:", e)
        return None

model = load('DigitRecognizer.keras')

st.header("Write a digit below:")

#x = st.checkbox("Tick to Draw, Untick to Delete", True)
dm = "freedraw" #if x else "transform"
s = 192
c = st_canvas(background_color = "#000000", fill_color = "#000000", stroke_color = "#ffffff", stroke_width = 15, height = s, width = s, drawing_mode = dm, key = "canvas")
if c.image_data is not None:
    i = cv2.resize(c.image_data.astype("uint8"), (28, 28))
    r = cv2.resize(i, (s, s), interpolation = cv2.INTER_NEAREST)
    st.header("Your Input Is:")
    st.image(r)

if st.button("PREDICT NOW"):
    X_test = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    z = X_test.reshape(1, 784)
    p = model.predict(z)
    pred = tf.nn.softmax(p)
    yhat = np.argmax(pred)
    st.success(f"The model predicts this to be **{yhat}**", icon = ":material/search:")
