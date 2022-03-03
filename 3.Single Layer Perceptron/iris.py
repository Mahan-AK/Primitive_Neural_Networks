from perceptron_copy import train_perceptron, predict
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

# for x in y:
#     print(x.index(1))

# exit()

X_train, X_test = X[:int(len(X)*(7.5/10))], X[int(len(X)*(7.5/10)):]
y_train, y_test = y[:int(len(y)*(7.5/10))], y[int(len(y)*(7.5/10)):]

alpha = 0.6
teta = 0.2

def act_func(y, teta):
        if y>teta: return 1
        if -1 * teta <= y <= teta: return 0
        else: return -1

w, b = train_perceptron(X_train, y_train, act_func, alpha, teta, max_iter=1000)

t = 0
for xs, ys in zip(X_test, y_test):
    if predict(xs, w, b) == ys:
        t += 1
    else:
        print(ys.index(1))


print(f"\nAccuracy (tested on {len(y_train)} samples): {t/len(X_test) * 100}%")