
import tensorflow as tf
import tensorflow_datasets as tfds

datos, metadatos = tfds.load('mnist', as_supervised=True, with_info=True)

datos_entrenamiento, datos_prueba = datos['train'], datos['test']

def normalizar(imagenes, etiquetas):
  imagenes = tf.cast(imagenes, tf.float32)
  imagenes /= 255
  return imagenes, etiquetas

datos_entrenamiento = datos_entrenamiento.map(normalizar)
datos_prueba = datos_prueba.map(normalizar)

modelo = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), input_shape=(28,28,1), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=100, activation='relu'),

    tf.keras.layers.Dense(50, activation='softmax')
])

modelo.summary()

modelo.compile(
    optimizer='adam',
    loss= tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

ej_entrenamiento = metadatos.splits["train"].num_examples
ej_prueba = metadatos.splits["test"].num_examples

tamano_lotes = 32

datos_entrenamiento = datos_entrenamiento.repeat().shuffle(ej_entrenamiento).batch(tamano_lotes)
datos_prueba = datos_prueba.batch(tamano_lotes)

import math

historial = modelo.fit(datos_entrenamiento, epochs=60, steps_per_epoch= math.ceil(ej_entrenamiento/tamano_lotes))

modelo.save('numeros_conv.h5')