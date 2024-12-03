import tensorflow as tf
import numpy as np
import os
import zipfile
import gdown
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Google Drive file ID
DATASET_FILE_ID = "1JlOdCrFv4icuK1JrvUhHy6weGiUSWKzp"
# Destination for downloaded dataset
DATASET_PATH = "Person_Photos.zip"
EXTRACT_PATH = "Person_Photos"

# Download the dataset using gdown
def download_dataset(file_id, destination):
    print("Descargando dataset...")
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, destination, quiet=False)
    print("Descarga completada.")

# Extract the ZIP file
def extract_dataset(zip_path, extract_path):
    print("Descomprimiendo dataset...")
    # Ensure extraction directory exists
    os.makedirs(extract_path, exist_ok=True)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Descompresión completada.")

# Function to load and preprocess images
def load_and_preprocess_images(directory, classes):
    images = []
    labels = []
    
    for i, class_name in enumerate(classes):
        class_path = os.path.join(directory, class_name)
        # Check if directory exists
        if not os.path.exists(class_path):
            print(f"Warning: Directory {class_path} does not exist!")
            continue
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                # Open image and convert to grayscale
                img = Image.open(img_path).convert('L')
                # Resize to 28x28 to match MNIST model
                img = img.resize((28, 28))
                # Convert to numpy array
                img_array = np.array(img)
                images.append(img_array)
                labels.append(i)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    return np.array(images), np.array(labels)

# Download and extract dataset
download_dataset(DATASET_FILE_ID, DATASET_PATH)
extract_dataset(DATASET_PATH, EXTRACT_PATH)

# Classes
classes = ["Enigma", "Nayelli"]

# Load images
X, y = load_and_preprocess_images(EXTRACT_PATH, classes)

# Check if we have images
if len(X) == 0:
    raise ValueError("No se encontraron imágenes. Verifica la ruta del dataset y los nombres de las clases.")

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape and normalize
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

# One-hot encode labels
Y_train = to_categorical(Y_train, num_classes=len(classes))
Y_test = to_categorical(Y_test, num_classes=len(classes))

# Data Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    zoom_range=[0.5,1.5]
)
datagen.fit(X_train)

# Model Architecture
modelo = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(len(classes), activation="softmax")
])

# Compilation
modelo.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Training with data augmentation
data_gen_entrenamiento = datagen.flow(X_train, Y_train, batch_size=32)

print("Entrenando modelo...")
epocas = 60
history = modelo.fit(
    data_gen_entrenamiento,
    epochs=epocas,
    validation_data=(X_test, Y_test),
    steps_per_epoch=int(np.ceil(X_train.shape[0] / float(32))),
    validation_steps=int(np.ceil(X_test.shape[0] / float(32)))
)

# Evaluation
test_loss, test_accuracy = modelo.evaluate(X_test, Y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Export for TensorFlow Serving
# Create a directory structure for TF Serving
export_path = 'faces-model/1'
os.makedirs(export_path, exist_ok=True)

# Create a concrete function for serving
@tf.function(input_signature=[tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32)])
def serve(input_data):
    return {"outputs": modelo(input_data)}

# Save the model with a concrete function for serving
tf.saved_model.save(modelo, export_path, signatures={
    'serving_default': serve
})

print(f"Model exported to {export_path}")

# Optional: Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
