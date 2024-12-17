import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import kagglehub

# this downloads the dataset from kaggle
# chosen dataset: Chest Xray Pneumonia photos
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("Path to dataset files:", path)

# sets seed
np.random.seed(42)

# paths for images in dataset
train_dir = f'{path}\\chest_xray\\train'
val_dir = f'{path}\\chest_xray\\val'
test_dir = f'{path}\\chest_xray\\test'

# preprocesses and auguments the data so i can analyze it.
# also makes generators for the training, val, and test sets
datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,horizontal_flip=True, fill_mode='nearest')

train_generator = datagen.flow_from_directory(train_dir,target_size=(150, 150),batch_size=32,class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(val_dir,target_size=(150, 150),batch_size=32,class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir,target_size=(150, 150),batch_size=32,class_mode='binary',shuffle=False)

# CNN model is built
cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# displays model architecture, compiles, and trains
cnn.summary()
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history = cnn.fit(train_generator,epochs=10,validation_data=val_generator)

# Evaluate the model
score = cnn.evaluate(test_generator)
print(f"Test Loss: {score[0]}\nTest Accuracy: {score[1]}")

# Visualize training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()

# Evaluate a balanced set of 20 test images and display predictions
test_generator.reset()

# Collect images and labels dynamically until we have a balanced set
normal_images, pneumonia_images = [], []
normal_labels, pneumonia_labels = [], []

for _ in range(len(test_generator)):  # Iterate through all batches
    batch_images, batch_labels = next(test_generator)
    for img, lbl in zip(batch_images, batch_labels):
        if lbl == 0 and len(normal_images) < 10:  # 'NORMAL' class
            normal_images.append(img)
            normal_labels.append(lbl)
        elif lbl == 1 and len(pneumonia_images) < 10:  # 'PNEUMONIA' class
            pneumonia_images.append(img)
            pneumonia_labels.append(lbl)
        if len(normal_images) == 10 and len(pneumonia_images) == 10:
            break
    if len(normal_images) == 10 and len(pneumonia_images) == 10:
        break

# Combine the two sets
selected_images = np.array(normal_images + pneumonia_images)
selected_labels = np.array(normal_labels + pneumonia_labels)

# Make predictions
predictions = cnn.predict(selected_images)
predicted_labels = ["Pneumonia" if pred > 0.5 else "Normal" for pred in predictions]

# Plot the selected images with predictions
plt.figure(figsize=(20, 10))
for i in range(len(selected_images)):
    plt.subplot(4, 5, i + 1)
    plt.imshow(selected_images[i])
    plt.title(f"Pred: {predicted_labels[i]}\nTrue: {'Pneumonia' if selected_labels[i] > 0.5 else 'Normal'}")
    plt.axis('off')
plt.tight_layout()
plt.show()
