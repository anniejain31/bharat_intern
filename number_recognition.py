# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.reshape((60000, 784))  # Flatten the images (28x28 pixels) into a single vector
X_test = X_test.reshape((10000, 784))
X_train = X_train.astype('float32') / 255.0  # Normalize pixel values between 0 and 1
X_test = X_test.astype('float32') / 255.0
y_train = to_categorical(y_train)  # Convert labels to one-hot encoded vectors
y_test = to_categorical(y_test)

# Build the neural network model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=1)

# Evaluate the model
_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy:', accuracy)
