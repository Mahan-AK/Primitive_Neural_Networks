from adaline import train_adaline, predict
from random import randint

def shuffle(X, y):
    for i in range(4):
        for index in range(len(X) - 1, -1, -1):
            swap_index = randint(index, len(X) - 1)

            X[index], X[swap_index] = X[swap_index], X[index]
            y[index], y[swap_index] = y[swap_index], y[index]

    return X, y

X = []
y = []

def encode(s):
    s = s.strip("\n")

    if s == "Iris-setosa": return [1, -1, -1].copy()
    elif s == "Iris-versicolor": return [-1, 1, -1].copy()
    elif s == "Iris-virginica": return [-1, -1, 1].copy()

with open("iris.txt", 'r') as f:
    f.readline()
    for line in f.readlines():
        t = line.split(',')
        X.append([float(t[0]), float(t[1]), float(t[2]), float(t[3])])
        y.append(encode(t[4]))

X, y = shuffle(X, y)

X_train, X_test = X[:int(len(X)*(7/10))], X[int(len(X)*(7/10)):]
y_train, y_test = y[:int(len(y)*(7/10))], y[int(len(y)*(7/10)):]

alpha = 0.02

def act_func(y):
    return 1 if y >= 0 else -1

w, b = train_adaline(X_train, y_train, alpha, max_iter=1000)

t = 0
for xs, ys in zip(X_test, y_test):
    if predict(xs, w, b, act_func) == ys:
        t += 1
    # else:
    #     print(predict(xs, w, b, act_func))
    #     print()

    

print(f"\nAccuracy (tested on {len(y_test)} samples): {t/len(X_test) * 100}%")
