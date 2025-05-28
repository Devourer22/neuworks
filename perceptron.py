import numpy as np

# Пороговая функция активации
def step_function(x):
    return 1 if x >= 0 else 0

# Класс Персептрона
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand(1)
        self.learning_rate = learning_rate

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return step_function(summation)

    def train(self, training_inputs, labels, epochs=10):
        for _ in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

# Эталонное изображение "1" (4x5)
reference_one = [
    0,0,1,0,
    0,1,1,0,
    0,0,1,0,
    0,0,1,0,
    0,0,1,0
]

# Искажённые варианты "1"
distorted_ones = [
    # Удалён последний пиксель
    [0,0,1,0,
     0,1,1,0,
     0,0,1,0,
     0,0,1,0,
     0,0,0,0],

    # Пиксель во втором ряду изменён
    [0,0,1,0,
     0,1,0,0,
     0,0,1,0,
     0,0,1,0,
     0,0,1,0],

    # Добавлен шум в первом ряду
    [1,0,1,0,
     0,1,1,0,
     0,0,1,0,
     0,0,1,0,
     0,0,1,0],

    # Добавлен шум в нижней части
    [0,0,1,0,
     0,1,1,0,
     0,0,1,1,
     0,0,1,0,
     0,0,1,1],

    # Половина изображения обнулена
    [0,0,1,0,
     0,1,1,0,
     0,0,0,0,
     0,0,0,0,
     0,0,0,0],
]

# Подготовка данных
training_inputs = np.array([reference_one] + distorted_ones)
labels = np.array([1] + [0] * len(distorted_ones))

# Создание и обучение персептрона
perceptron = Perceptron(input_size=20)
perceptron.train(training_inputs, labels, epochs=20)

# Проверка на тех же входах
for idx, test_input in enumerate(training_inputs):
    prediction = perceptron.predict(test_input)
    print(f"Тест {idx+1}: Предсказание = {prediction}, Ожидание = {labels[idx]}")
