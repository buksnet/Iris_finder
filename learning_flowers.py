import random as rnd  # перемешивание данных и получение случайных значений
import matplotlib.pyplot as plt  # построение графиков
import numpy as np  # выполнение математических преобразований
from sklearn import datasets  # выборки данных

iris = datasets.load_iris()

dataset = [(iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]  # получение выборки данных

INPUT_DIM = 4  # число входов
OUTPUT_DIM = 3  # число выходов
H_DIM = 10  # число спрятанных слоёв
EPOCHS = 500  # число итераций (эпох)
BATCH_SIZE = 50  # размер охапки (batch-a)

w1 = np.random.randn(INPUT_DIM, H_DIM)  # получение случайной матрицы весов
b1 = np.random.randn(1, H_DIM)  # получение вектора случайных коэффициентов

w2 = np.random.randn(H_DIM, OUTPUT_DIM)  # получение случайной матрицы весов
b2 = np.random.randn(OUTPUT_DIM)  # получение вектора случайных коэффициентов

w1 = (w1 - 0.5) * 2 * np.sqrt(1 / INPUT_DIM)  # преобразования над данными для увеличения точности сети
b1 = (b1 - 0.5) * 2 * np.sqrt(1 / INPUT_DIM)
w2 = (w2 - 0.5) * 2 * np.sqrt(1 / INPUT_DIM)
b2 = (b2 - 0.5) * 2 * np.sqrt(1 / INPUT_DIM)

loss_arr = list()  # список вычисленных ошибок

ALPHA = 0.0002  # параметр скорости обучения (выше скорость - ниже точность)


def sparse_cross_entropy(z, y):  # Функция вычисления ошибки
    return -np.log(z[0, y])


def sparse_cross_entropy_batch(z, y):  # Функция вычисления ошибки для охапки данных (batch-ей)
    return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))


def relu(t):  # активационная функция (если переданное значение > 0, то возвращает его, иначе возвращает 0)
    return np.maximum(t, 0)


def softmax(h):
    out = np.exp(h)  # получение экспоненты
    return out / np.sum(out)  # находим сумму экспонент всех эл-тов


def softmax_batch(t):
    out = np.exp(t)
    return out / np.sum(out, axis=1, keepdims=True)  # находим сумму экспонент всех эл-тов


def to_full(y, num_classes):  # получение матрицы ошибок
    y_full = np.zeros((1, num_classes))
    y_full[0, y] = 1
    return y_full


def to_full_batch(y, num_classes):  # получение матрицы ошибок
    y_full = np.zeros((len(y), num_classes))
    for j, yj in enumerate(y):
        y_full[j, yj] = 1
    return y_full


def relu_deriv(t):  # активационная функция, но возвращает число с плавающей точкой
    return (t >= 0).astype(float)


for epoch in range(EPOCHS):
    rnd.shuffle(dataset)
    for i in range(len(dataset) // BATCH_SIZE):
        batch_x, batch_y = zip(*dataset[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE])  # предварительная обработка данных
        x = np.concatenate(batch_x, axis=0)
        y = np.array(batch_y)

        # Прямое распространение (вычисления по формулам)
        t1 = x @ w1 + b1
        h1 = relu(t1)
        t2 = h1 @ w2 + b2
        z = softmax_batch(t2)
        E = np.sum(sparse_cross_entropy_batch(z, y))  # спарс потому что y не вектор распределения, а индекс класса

        # Обратное распространение (вычисления по формулам)
        y_full = to_full_batch(y, OUTPUT_DIM)
        dE_dt2 = z - y_full
        dE_dw2 = h1.T @ dE_dt2
        dE_db2 = np.sum(dE_dt2, axis=0, keepdims=True)
        dE_dh1 = dE_dt2 @ w2.T
        dE_dt1 = dE_dh1 * relu_deriv(t1)
        dE_dw1 = x.T @ dE_dt1
        dE_db1 = np.sum(dE_dt1, axis=0, keepdims=True)

        # Обновление весов (вычисления по формулам)
        w1 = w1 - ALPHA * dE_dw1  # это матрицы одинакового размера
        b1 = b1 - ALPHA * dE_db1
        w2 = w2 - ALPHA * dE_dw2
        b2 = b2 - ALPHA * dE_db2

        loss_arr.append(E)


def predict(x):  # получение предсказания нейросети
    t1 = x @ w1 + b1
    h1 = relu(t1)
    t2 = h1 @ w2 + b2
    z = softmax(t2)
    return z


def calc_accuracy():
    correct = 0
    for x, y in dataset:
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct += 1
    return correct / len(dataset)


accuracy = calc_accuracy()
print("Accuracy: ", round(accuracy * 100), "%", sep="")

if accuracy >= 0.85:
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    test_data = []

    z = predict(np.array([7.9, 3.1, 7.5, 1.8]))

    y_pred = np.argmax(z)
    print('Predicted class:', class_names[y_pred])

    z = predict(np.array([8.0, 3.0, 7.4, 1.7]))

    y_pred = np.argmax(z)
    print('Predicted class:', class_names[y_pred])

    z = predict(np.array([8.1, 2.8, 7.3, 1.6]))

    y_pred = np.argmax(z)
    print('Predicted class:', class_names[y_pred])
else:
    print("Слишком низкая точность, перезапустите.")

fig = plt.plot(loss_arr)
plt.ylim(0, 100)
plt.suptitle("График ошибок")
plt.ylabel("Количество ошибок (в %)")
plt.xlabel("Число поколений")
plt.show()
