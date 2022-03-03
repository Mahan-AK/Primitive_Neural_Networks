import random

def diff2D(a, b):
    return [[q-w for q,w in zip(x,y)] for x,y in zip(a, b)]

def copy2D(a):
    return [[x for x in y] for y in a]

def abs2D(a):
    return [[abs(x) for x in y] for y in a]

def max2D(a):
    return max([max(x) for x in a])

def train_adaline(x_train, y_train, alpha, max_iter = 10000, epsilon = 0.001):
    input_layer_num = len(x_train[0])
    output_layer_num = len(y_train[0])
    train_samples = len(x_train)

    weights = [[random.random()/2-0.25 for _ in range(output_layer_num)] for _ in range(input_layer_num)]
    biases = [random.random()/2-0.25 for _ in range(output_layer_num)]

    iter = 0
    converge = False

    while True:
        last_w = copy2D(weights)

        for i in range(train_samples):
            for k in range(output_layer_num):
                yIN = sum([weights[z][k] * x_train[i][z] for z in range(input_layer_num)]) + biases[k]

                for j in range(input_layer_num):
                    weights[j][k] += alpha * (y_train[i][k] - yIN) * x_train[i][j]

                biases[k] += alpha * (y_train[i][k] - yIN)

        if max2D(abs2D(diff2D(weights, last_w))) < epsilon:
            converge = True
            break

        iter += 1

        if max_iter and iter >= max_iter: break
        print(iter, end='\r')

    if converge:
        print(f"Convergence in {iter} epochs with epsilon = {epsilon}\n")
    else:
        print(f"Max iter of {iter} reached without convergence (epsilon = {epsilon}) \n")
                
    return weights, biases

def predict(inp, weights, biases, f):
    out = [0 for _ in range(len(biases))]

    for i in range(len(biases)):
        for j in range(len(inp)):
            out[i] += inp[j] * weights[j][i]
        out[i] += biases[i]

    return [f(o) for o in out]
