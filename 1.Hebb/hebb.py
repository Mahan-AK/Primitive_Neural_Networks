def train_hebb(x_train, y_train):
    input_layer_num = len(x_train[0])
    output_layer_num = len(y_train[0])
    train_samples = len(x_train)

    weights = [[0 for _ in range(output_layer_num)] for _ in range(input_layer_num)]
    biases = [0 for _ in range(output_layer_num)]

    for i in range(train_samples):
        for k in range(output_layer_num):
            for j in range(input_layer_num):
                weights[j][k] += x_train[i][j] * y_train[i][k]
            biases[k] += y_train[i][k]

    return weights, biases

def predict(inp, weights, biases, f):
    out = [0 for _ in range(len(biases))]

    for i in range(len(biases)):
        for j in range(len(inp)):
            out[i] += inp[j] * weights[j][i]
        out[i] += biases[i]

    return [f(o) for o in out]