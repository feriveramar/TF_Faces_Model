import tensorflow as tf
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define el tamaño de las imágenes y las clases
TAMANO_IMG = 128
mi_clases = ['Enigma', 'Nayelli']  # Cambia esto con los nombres de tus clases

# Función para cargar y preprocesar imágenes
def load_and_preprocess_image(file_path, label):
    file_path = tf.cast(file_path, tf.string)
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)  # Usa channels=3 para imágenes en color
    img = tf.image.resize(img, [TAMANO_IMG, TAMANO_IMG])
    img = img / 255.0  # Normaliza a [0, 1]
    return img, label

# Cargar rutas de imágenes y etiquetas
image_paths = []
labels = []
base_dir = 'caras_fotos'  # Directorio base después de extraer el dataset

for i, mi_clase in enumerate(mi_clases):
    class_dir = os.path.join(base_dir, mi_clase)
    if not os.path.exists(class_dir):
        print(f"No se encontró la carpeta: {class_dir}")
    else:
        for img_path in os.listdir(class_dir):
            full_path = os.path.join(class_dir, img_path)
            image_paths.append(full_path)
            labels.append(i)

dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(load_and_preprocess_image)

# Convertir el dataset a listas y luego a arrays numpy
dataset_list = list(dataset)
X = np.array([img.numpy() for img, _ in dataset_list])
y = np.array([label.numpy() for _, label in dataset_list])

# Convertir las etiquetas a categorías
y = tf.keras.utils.to_categorical(y, num_classes=len(mi_clases))

print(X.shape, y.shape)

# Dividir los datos
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aumento de datos
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    zoom_range=[0.5, 1.5]
)
datagen.fit(X_train)

# Arquitectura del modelo
modelo = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(TAMANO_IMG, TAMANO_IMG, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(len(mi_clases), activation='softmax')
])

# Compilación del modelo
modelo.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Entrenamiento con aumento de datos
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

# Evaluación
test_loss, test_accuracy = modelo.evaluate(X_test, Y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Exportar para TensorFlow Serving
export_dir = 'faces-model/1'
os.makedirs(export_dir, exist_ok=True)

# Define la función de servicio
@tf.function(input_signature=[tf.TensorSpec(shape=(None, TAMANO_IMG, TAMANO_IMG, 3), dtype=tf.float32)])
def serve(input_data):
    return {"outputs": modelo(input_data)}

# Guarda el modelo
tf.saved_model.save(modelo, export_dir, signatures={'serving_default': serve})

print(f"\nModel saved to: {export_dir}")
print("Contents of the export directory:")
for root, dirs, files in os.walk(export_dir):
    for file in files:
        print(os.path.join(root, file))

# Guarda los nombres de las clases en un archivo
with open(os.path.join(export_dir, 'class_names.txt'), 'w') as f:
    for cls in flower_classes:
        f.write(f"{cls}\n")

# Gráfica de la historia de entrenamiento
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
