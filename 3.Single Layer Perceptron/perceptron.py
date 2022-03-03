def train_perceptron(x_train, y_train, act_func, alpha, teta, max_iter = None):
    input_layer_num = len(x_train[0])
    output_layer_num = len(y_train[0])
    train_samples = len(x_train)

    weights = [[0 for _ in range(output_layer_num)] for _ in range(input_layer_num)]
    biases = [0 for _ in range(output_layer_num)]

    update = True
    iter = 0

    while update:
        update = False

        for i in range(train_samples):
            for k in range(output_layer_num):
                yIN = sum([weights[z][k] * x_train[i][z] for z in range(input_layer_num)]) + biases[k]
                y = act_func(yIN, teta)

                if y != y_train[i][k]:
                    update = True
                    for j in range(input_layer_num):
                        weights[j][k] += alpha * x_train[i][j] * y_train[i][k]
                        
                    biases[k] += alpha * y_train[i][k]

        iter += 1

        if max_iter and iter >= max_iter: break
        print(iter, end='\r')

    print(f"Iteration count: {iter}\n")
                
    return weights, biases

def predict(inp, weights, biases, f, teta):
    out = [0 for _ in range(len(biases))]

    for i in range(len(biases)):
        for j in range(len(inp)):
            out[i] += inp[j] * weights[j][i]
        out[i] += biases[i]

    return [f(o, teta) for o in out]
