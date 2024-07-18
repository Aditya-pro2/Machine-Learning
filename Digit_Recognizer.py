import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.datasets import mnist

class StreamlitProgressCallback(Callback):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.progress((epoch + 1) / self.epochs)
        self.status_text.text(f'Epoch {epoch + 1}/{self.epochs}, Loss: {logs["loss"]:.4f}')

st.title(':blue[DIGIT RECOGNIZER]')
# Load Data
st.header('Load Data (MNIST Dataset)')
if st.button('Load'):
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    st.write("Shape of X_train:", X_train.shape)
    st.write("Shape of Y_train:", Y_train.shape)
    st.write("Shape of X_test:", X_test.shape)
    st.write("Shape of Y_test:", Y_test.shape)
    st.header('Training and Testing Data Reshaped')
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    st.write("Shape of X_train:", X_train.shape)
    st.write("Shape of X_test:", X_test.shape)
    
    # Display a few images
    st.header('Display a Few Images')
    fig, axes = plt.subplots(8, 8, figsize=(5, 5))
    fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91])
    for i, ax in enumerate(axes.flat):
        p = np.random.randint(X_train.shape[0])
        ax.imshow(X_train[p].reshape(28, 28), cmap = 'gray')
        ax.set_title(f"Label: {Y_train[p]}", fontsize = 7)
        ax.axis('off')
    st.pyplot(fig)
    
    tf.random.set_seed(123)
    model = Sequential(
        [
            tf.keras.Input(shape = (784,)),
            Dense(units = 25, activation = 'relu', name = "L1"),
            Dense(units = 15, activation = 'relu', name = "L2"),
            Dense(units = 10, activation = 'linear', name = "L3")
        ], name = "Digit_Recognizer"
    )
    st.header('Model Summary')
    model.summary(print_fn = lambda x: st.text(x))
    st.header('Training Progress')
    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))
    callback = StreamlitProgressCallback(40)
    model.fit(X_train, Y_train, epochs = 40, callbacks = [callback])
    pred = model.predict(X_test)
    p = tf.nn.softmax(pred)
    yhat = [np.argmax(i) for i in p]
    st.header('Test Set Report')
    r = sum(yhat == Y_test)
    st.write("Accuracy:", r / 100, "%")
    data = pd.DataFrame({"Names": ["Correct", "Wrong"], "Values": [r, len(yhat) - r]})
    d = data.set_index("Names")
    st.bar_chart(d, horizontal = True)
