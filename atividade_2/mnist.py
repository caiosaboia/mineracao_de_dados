import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random

# 1. Carregar o conjunto de dados MNIST
data = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()

# 2. Normalizar os dados (0-255 -> 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. Definir a arquitetura da rede neural
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Achatar imagens 28x28 para vetor 784
    keras.layers.Dense(128, activation='relu'),  # Camada oculta com 128 neurônios
    keras.layers.Dense(10, activation='softmax') # Camada de saída para 10 classes
])

# 4. Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Treinar o modelo
epochs = 20  # Número de épocas
history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

# 6. Avaliação no conjunto de teste
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Perda: {loss:.4f}, Acurácia: {accuracy:.4f}')

# 7. Exemplo de previsão
predictions = model.predict(x_test)
plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.title(f'Predição: {np.argmax(predictions[0])}')
plt.show()

# 8. Visualizar erros do modelo
# 9. Sortear várias imagens aleatórias e verificar as predições
random_indices = random.sample(range(len(x_test)), 10)
predictions_list = [np.argmax(predictions[idx]) for idx in random_indices]
real_values_list = [y_test[idx] for idx in random_indices]

for i, idx in enumerate(random_indices):
    if predictions_list[i] == real_values_list[i]:
        print(f"acertou")
    else:
        print(f"errou")
