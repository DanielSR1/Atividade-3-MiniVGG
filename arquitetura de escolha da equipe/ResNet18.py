import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from keras import backend as K
from keras import utils as utls
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam as adam
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

# Normaliza os dados
XTreino = XTreino / 255.0
XTeste = XTeste / 255.0

def resnet_block(x, filters, kernel_size=3, stride=1, downsample=False):
    shortcut = x

    # Convolução 3x3
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Convolução 3x3
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Downsampling (se necessário) para garantir que o atalho e a saída de x tenham as mesmas dimensões
    if downsample:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Adicionando o atalho
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x

def build_resnet18():
    inputs = layers.Input(shape=(32, 32, 3))

    # Primeira camada convolucional
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # ResNet Block 1
    x = resnet_block(x, 64, downsample=False)

    # ResNet Block 2
    x = resnet_block(x, 128, stride=2, downsample=True)

    # ResNet Block 3
    x = resnet_block(x, 256, stride=2, downsample=True)

    # ResNet Block 4
    x = resnet_block(x, 512, stride=2, downsample=True)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Camada densa para classificação
    x = layers.Dense(10, activation='softmax')(x)

    # Criando o modelo
    model = models.Model(inputs, x)

    return model


modelo = build_resnet18()

modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

modelo.summary()

modelo.fit(XTreino, yTreino, batchSize, epochs, validation_data=(XTeste, yTeste))

NomeDosRotulos = ["avião", "carro", "pássaro", "gato", "cervo", "cachorro", "sapo", "cavalo", "navio", "caminhão"]
perda_teste, acuracia_teste = modelo.evaluate(XTeste, yTeste, verbose=2)
print(f'Acurácia no conjunto de teste: {acuracia_teste * 100:.2f}%')

nomeDosRotulos = ["avião", "carro", "pássaro", "gato", "cervo", "cachorro", "sapo", "cavalo", "navio", "caminhão"]
predicao = modelo.predict(XTeste)
print(classification_report(yTeste.argmax(axis=1), predicao.argmax(axis=1), target_names=nomeDosRotulos))

class_names = ["avião", "carro", "pássaro", "gato", "cervo", "cachorro", "sapo", "cavalo", "navio", "caminhão"]
yPred = modelo.predict(XTeste)
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