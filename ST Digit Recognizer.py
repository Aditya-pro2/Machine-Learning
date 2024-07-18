import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import mnist
from PIL import Image

st.title('Digit Recognizer')

# Load Data
st.header('Load Data:')
if st.button('Load'):
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    st.write("Shape of X_train: ", X_train.shape)
    st.write("Shape of Y_train: ", Y_train.shape)
    st.write("Shape of X_test: ", X_test.shape)
    st.write("Shape of Y_test: ", Y_test.shape)
    st.header('Training and Testing Data Reshaped.')
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    st.write("Shape of X_train: ", X_train.shape)
    st.write("Shape of X_test: ", X_test.shape)

# Display a few images
fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    axes[i].imshow(X_train[i], cmap='gray')
    axes[i].set_title(f"Label: {y_train[i]}")
    axes[i].axis('off')
st.pyplot(fig)

# Define the model
st.header('Define Model:')
tf.random.set_seed(123)
model = Sequential(
    [
        tf.keras.Input(shape = (784,)),
        Dense(units = 25, activation = 'relu', name = "L1"),
        Dense(units = 15, activation = 'relu', name = "L2"),
        Dense(units = 10, activation = 'linear', name = "L3")
    ], name = "digit_recognizer"
)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary(print_fn=lambda x: st.text(x))

# Define the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath='best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    save_weights_only=False,
    verbose=1
)

# Train the model
st.header('Train Model')
if st.button('Train'):
    history = model.fit(
        X_train, y_train,
        epochs=10,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint_callback]
    )

    # Plot training history
    st.header('Training History')
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='Training Accuracy')
    ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax.legend()
    st.pyplot(fig)

# Load the best model
st.header('Load Best Model')
if st.button('Load Best Model'):
    model.load_weights('best_model.h5')
    st.success('Best model loaded successfully.')

# Predict on user-uploaded images
st.header('Predict on Uploaded Image')
uploaded_file = st.file_uploader("Upload an image of a digit", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image).reshape(1, 28, 28)

    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction, axis=1)[0]
    st.write(f"Predicted Digit: {predicted_digit}")
