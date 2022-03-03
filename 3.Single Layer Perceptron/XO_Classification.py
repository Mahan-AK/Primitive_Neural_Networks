import random
from sys import argv
from perceptron import train_perceptron, predict

def get_variant_sample(l):
    variance = random.choice([0, 1, 2])
    flip = random.sample(l, k=len(l)-variance)
    sample = [-1] * 25

    for f in flip:
        sample[f] = 1

    return sample

if __name__ == "__main__":    
    X_indexes = [0, 4, 6, 8, 12, 16, 18, 20, 24]
    O_indexes = [1, 2, 3, 5, 9, 10, 14, 15, 19, 21, 22, 23]

    x_train = [get_variant_sample(X_indexes) for _ in range(20)] + [get_variant_sample(O_indexes) for _ in range(20)]
    y_train = [[1]] * 20 + [[-1]] * 20

    def act_func(y, teta):
        if y>teta: return 1
        if -1 * teta <= y <= teta: return 0
        else: return -1

    if len(argv) < 2:
        alpha, teta = 0.6, 0.2
    else:
        alpha = float(argv[1])
        teta = float(argv[2])

    w, b = train_perceptron(x_train, y_train, act_func, alpha, teta)

    print(f"Weights: {w}")
    print(f"Biases: {b}")
    print()

    test = get_variant_sample(random.choice([X_indexes, O_indexes]))
    for i in range(5):
        for j in range(5):
            print("*" if test[i*5 + j] == 1 else " ", end='')
        print()

    print(f'Prediction for sample: {"X" if predict(test, w, b, act_func, teta)[0] == 1 else "O"}')

    t = 0
    for _ in range(500):
        test = get_variant_sample(X_indexes)
        if predict(test, w, b, act_func, teta)[0] == 1:
            t += 1

    for _ in range(500):
        test = get_variant_sample(O_indexes)
        if predict(test, w, b, act_func, teta)[0] == -1:
            t += 1

    print(f"\nAccuracy (tested on 1000 samples): {t/1000 * 100}%")
 