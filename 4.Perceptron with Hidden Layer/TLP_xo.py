import random
import math
from TLP import train_TLP, predict

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
    
    # using bipolar sigmoid function
    def act(x):
        return math.tanh(x/2)

    # derivitive of f:
    def der_act(x):
        return 1/(math.cosh(x) + 1)

    w_h, b_h, w_o, b_o = train_TLP(x_train, y_train, hidden_layer_num=4, alpha=0.01, f=act, df=der_act)

    print("Hidden Layer:")
    print(f"Weights: {w_h}")
    print(f"Biases: {b_h}")
    print()

    print("Output Layer:")
    print(f"Weights: {w_o}")
    print(f"Biases: {b_o}")
    print()

    quantizer = lambda x: 1 if x >= 0 else -1

    test = get_variant_sample(random.choice([X_indexes, O_indexes]))
    for i in range(5):
        for j in range(5):
            print("*" if test[i*5 + j] == 1 else " ", end='')
        print()

    print(f'Prediction for sample: {"X" if predict(test, w_h, b_h, w_o, b_o, act, quantizer)[0] == 1 else "O"}')

    t = 0
    for _ in range(500):
        test = get_variant_sample(X_indexes)
        if predict(test, w_h, b_h, w_o, b_o, act, quantizer)[0] == 1:
            t += 1

    for _ in range(500):
        test = get_variant_sample(O_indexes)
        if predict(test, w_h, b_h, w_o, b_o, act, quantizer)[0] == -1:
            t += 1

    print(f"\nAccuracy (tested on 1000 samples): {t/1000 * 100}%")
 