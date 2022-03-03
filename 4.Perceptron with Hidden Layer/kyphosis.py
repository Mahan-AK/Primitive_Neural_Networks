from TLP import train_TLP, predict, print_confusion_matrix
from random import randint
import math

def shuffle(X, y):
    for _ in range(4):
        for index in range(len(X) - 1, -1, -1):
            swap_index = randint(index, len(X) - 1)

            X[index], X[swap_index] = X[swap_index], X[index]
            y[index], y[swap_index] = y[swap_index], y[index]

    return X, y

X = []
y = []

def encode(s):
    if s == "\"present\"": return [1].copy()
    elif s == "\"absent\"": return [-1].copy()

with open("kyphosis.csv", 'r') as f:
    f.readline()
    for line in f.readlines():
        t = line.strip('\n').split(',')
        X.append(list(map(int, t[1:])))
        y.append(encode(t[0]))

X, y = shuffle(X, y)

X_train, X_test = X[:int(len(X)*(9/10))], X[int(len(X)*(9/10)):]
y_train, y_test = y[:int(len(y)*(9/10))], y[int(len(y)*(9/10)):]

# using bipolar sigmoid function
def act(x):
    return math.tanh(x/2)

# derivitive of f:
def der_act(x):
    return 1/(math.cosh(x) + 1)

w_h, b_h, w_o, b_o = train_TLP(X_train, y_train, hidden_layer_num=4, alpha=0.033, f=act, df=der_act)

print("Hidden Layer:")
print(f"Weights: {w_h}")
print(f"Biases: {b_h}")
print()

print("Output Layer:")
print(f"Weights: {w_o}")
print(f"Biases: {b_o}")
print()

pred = predict(X_test, w_h, b_h, w_o, b_o)

t = sum([xs == ys for xs, ys in zip(pred, y_test)])

print(f"Accuracy (tested on {len(y_test)} samples): {t/len(X_test) * 100:.3f}%")

# print("\nConfusion matrix:")
# print_confusion_matrix(pred, y_test)