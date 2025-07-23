import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
def recognize_digit(image_dict):
    image = image_dict["composite"]
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    if image.shape[-1] == 3:
        image = np.mean(image, axis=-1)
    image = tf.image.resize(image[..., np.newaxis], [28, 28]).numpy()
    image = 1 - (image.astype('float32') / 255.0)
    image = image.reshape(1, 28, 28, 1)
    prediction = model.predict(image).tolist()[0]
    return {str(i): prediction[i] for i in range(10)}
gr.Interface(
    fn=recognize_digit,
    inputs="sketchpad",
    outputs=gr.Label(num_top_classes=3),
    live=True,
    title="MNIST Digit Recognizer",
    description="Draw a digit (0-9) below and watch the AI guess it!"
).launch(debug=True)