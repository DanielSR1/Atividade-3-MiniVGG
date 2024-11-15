#MiniVGG
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from keras import backend as K
from keras import utils as utls
from tensorflow.keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Definição de Hiperparâmetros
imageRows, imageCols, cores = 32, 32, 3
batchSize = 64
numClasses = 10
epochs = 5

# Carrega o dataset CIFAR-10
(XTreino, yTreino), (XTeste, yTeste) = cifar10.load_data()

# Normaliza os dados
XTreino = XTreino / 255.0
XTeste = XTeste / 255.0
yTreino = utls.to_categorical(yTreino, numClasses)
yTeste = utls.to_categorical(yTeste, numClasses)

XTreino.shape

def build_minivggnet(input_shape=(32, 32, 3), num_classes=10, dropout_rate=0.5):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Activation('relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(Activation('relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

num_classes = 10
input_shape = (32, 32, 3)

dropout_rate = 0.5  # Taxa de Dropout de 50%
model = build_minivggnet(input_shape=input_shape, num_classes=num_classes, dropout_rate=dropout_rate)

model.summary()

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

minivggmodel = model.fit(XTreino, yTreino, batch_size=batchSize, epochs=epochs, validation_data=(XTeste, yTeste))

nomeDosRotulos = ["avião", "carro", "pássaro", "gato", "cervo", "cachorro", "sapo", "cavalo", "navio", "caminhão"]
predicao = model.predict(XTeste)
print(classification_report(yTeste.argmax(axis=1), predicao.argmax(axis=1), target_names=nomeDosRotulos))

f, ax = plt.subplots()
ax.plot(minivggmodel.history['accuracy'], 'o-')
ax.plot(minivggmodel.history['val_accuracy'], 'x-')
ax.legend(['Acurácia no Treinamento', 'Acurácia na Validação'], loc=0)
ax.set_title('Treinamento e Validação - Acurácia por Época')
ax.set_xlabel('Época')
ax.set_ylabel('Acurácia')

class_names = ["avião", "carro", "pássaro", "gato", "cervo", "cachorro", "sapo", "cavalo", "navio", "caminhão"]
yPred = model.predict(XTeste)
yPred_Classe = np.argmax(yPred, axis=1)
yTrue = np.argmax(yTeste, axis=1)

cm = confusion_matrix(yTrue, yPred_Classe)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - CIFAR-10')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()