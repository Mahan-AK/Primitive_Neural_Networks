import random
from sys import argv
from adaline import train_adaline, predict

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

    def act_func(y):
        return 1 if y >= 0 else -1

    alpha = 0.01
    w, b = train_adaline(x_train, y_train, alpha)

    print(f"Weights: {w}")
    print(f"Biases: {b}")
    print()

    test = get_variant_sample(random.choice([X_indexes, O_indexes]))
    for i in range(5):
        for j in range(5):
            print("*" if test[i*5 + j] == 1 else " ", end='')
        print()

    print(f'Prediction for sample: {"X" if predict(test, w, b, act_func)[0] == 1 else "O"}')

    t = 0
    for _ in range(500):
        test = get_variant_sample(X_indexes)
        if predict(test, w, b, act_func)[0] == 1:
            t += 1

    for _ in range(500):
        test = get_variant_sample(O_indexes)
        if predict(test, w, b, act_func)[0] == -1:
            t += 1

    print(f"\nAccuracy (tested on 1000 samples): {t/1000 * 100}%")
 