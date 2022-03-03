import random

def train_TLP_valid(x_train, y_train, X_valid, y_valid, hidden_layer_num, alpha, f, df, tol = 10, max_iter = 1000):
    input_layer_num = len(x_train[0])
    output_layer_num = len(y_train[0])
    train_samples = len(x_train)

    hidden_layer = [0 for _ in range(hidden_layer_num)]
    hidden_layer_NI = [0 for _ in range(hidden_layer_num)]
    output_delta = [0 for _ in range(output_layer_num)]

    # initializing weights and biases with a small random number between -0.25 and 0.25
    weights_hidden = [[random.random()/2-0.25 for _ in range(input_layer_num)] for _ in range(hidden_layer_num)]
    weights_output = [[random.random()/2-0.25 for _ in range(hidden_layer_num)] for _ in range(output_layer_num)]
    biases_hidden = [random.random()/2-0.25 for _ in range(hidden_layer_num)]
    biases_output = [random.random()/2-0.25 for _ in range(output_layer_num)]

    itered = 0
    n_ovf = 0
    last_best = len(X_valid)
    best_params = weights_hidden, biases_hidden, weights_output, biases_output

    # training network
    while True:
        for i in range(train_samples):
            # calclulating output of hidden layer
            for j in range(hidden_layer_num):
                zNI = sum([weights_hidden[j][z] * x_train[i][z] for z in range(input_layer_num)]) + biases_hidden[j]
                hidden_layer_NI[j] = zNI
                hidden_layer[j] = f(zNI)

            # finding delta values for output layer and tweaking input weights and biases of output layer neurons
            for j in range(output_layer_num):
                yNI = sum([weights_output[j][z] * hidden_layer[z] for z in range(hidden_layer_num)]) + biases_output[j]
                delta = (y_train[i][j] - f(yNI)) * df(yNI)
                output_delta[j] = delta

                for k in range(hidden_layer_num):
                    weights_output[j][k] += alpha * delta * hidden_layer[k]

                biases_output[j] += alpha * delta

            # tweaking input weights and biases of hidden layer neurons with delta values of output layer
            for j in range(hidden_layer_num):
                D = sum([weights_output[z][j] * output_delta[z] for z in range(output_layer_num)])
                delta = D * df(hidden_layer_NI[j])

                for k in range(input_layer_num):
                    weights_hidden[j][k] += alpha * delta * x_train[i][k]

                biases_hidden[j] += alpha * delta

        itered += 1
        print(itered, end='\r')

        # if loss increases on validation dataset {tol} times, stop training 
        loss = _loss_on_df(X_valid, y_valid, weights_hidden, biases_hidden, weights_output, biases_output)

        if loss > last_best: n_ovf += 1
        else:
            last_best = loss
            best_params = weights_hidden, biases_hidden, weights_output, biases_output

        if n_ovf >= tol or itered >= max_iter: break

    print(f"Convergance in {itered} epochs.\n")

    return best_params

def train_TLP(x_train, y_train, hidden_layer_num, alpha, f, df):
    input_layer_num = len(x_train[0])
    output_layer_num = len(y_train[0])
    train_samples = len(x_train)

    hidden_layer = [0 for _ in range(hidden_layer_num)]
    hidden_layer_NI = [0 for _ in range(hidden_layer_num)]
    output_delta = [0 for _ in range(output_layer_num)]

    # initializing weights and biases with a small random number between -0.25 and 0.25
    weights_hidden = [[random.random()/2-0.25 for _ in range(input_layer_num)] for _ in range(hidden_layer_num)]
    weights_output = [[random.random()/2-0.25 for _ in range(hidden_layer_num)] for _ in range(output_layer_num)]
    biases_hidden = [random.random()/2-0.25 for _ in range(hidden_layer_num)]
    biases_output = [random.random()/2-0.25 for _ in range(output_layer_num)]

    # training network
    for itered in range(2000):
        for i in range(train_samples):
            # calclulating output of hidden layer
            for j in range(hidden_layer_num):
                zNI = sum([weights_hidden[j][z] * x_train[i][z] for z in range(input_layer_num)]) + biases_hidden[j]
                hidden_layer_NI[j] = zNI
                hidden_layer[j] = f(zNI)

            # finding delta values for output layer and tweaking input weights and biases of output layer neurons
            for j in range(output_layer_num):
                yNI = sum([weights_output[j][z] * hidden_layer[z] for z in range(hidden_layer_num)]) + biases_output[j]
                delta = (y_train[i][j] - f(yNI)) * df(yNI)
                output_delta[j] = delta

                for k in range(hidden_layer_num):
                    weights_output[j][k] += alpha * delta * hidden_layer[k]

                biases_output[j] += alpha * delta

            # tweaking input weights and biases of hidden layer neurons with delta values of output layer
            for j in range(hidden_layer_num):
                D = sum([weights_output[z][j] * output_delta[z] for z in range(output_layer_num)])
                delta = D * df(hidden_layer_NI[j])

                for k in range(input_layer_num):
                    weights_hidden[j][k] += alpha * delta * x_train[i][k]

                biases_hidden[j] += alpha * delta

        print(itered, end='\r')

    return weights_hidden, biases_hidden, weights_output, biases_output

def _loss_on_df(X_valid, y_valid, w_h, b_h, w_o, b_o):
    return sum([_pred(xs, w_h, b_h, w_o, b_o) != ys for xs, ys in zip(X_valid, y_valid)])

def _pred(X, weights_hidden, biases_hidden, weights_output, biases_output):
    hidden_layer = [0 for _ in range(len(biases_hidden))]

    for i in range(len(hidden_layer)):
        for j in range(len(X)):
            hidden_layer[i] += X[j] * weights_hidden[i][j]
        hidden_layer[i] += biases_hidden[i]

    out = [0 for _ in range(len(biases_output))]

    for i in range(len(out)):
        for j in range(len(hidden_layer)):
            out[i] += hidden_layer[j] * weights_output[i][j]
        out[i] += biases_output[i]

    return [1 if x == max(out) else -1 for x in out]

def predict(X, weights_hidden, biases_hidden, weights_output, biases_output):
    return [_pred(xs, weights_hidden, biases_hidden, weights_output, biases_output) for xs in X]

def confusion_matrix(pred, y_test):
    num_out = len(y_test[0])
    mat = [[0 for _ in range(num_out+1)] for _ in range(num_out+1)]

    for p, y in zip(pred, y_test):
        i_p = p.index(1)
        i_y = y.index(1)

        mat[i_y][i_p] += 1

    for i in range(num_out):
        try:
            mat[i][-1] = round(mat[i][i] / sum(mat[i][:-1]), 3)
        except:
            mat[i][-1] = 0.0
        try:
            mat[-1][i] = round(mat[i][i] / sum([mat[j][i] for j in range(num_out)]), 3)
        except:
            mat[-1][i] = 0.0
    
    try:
        mat[-1][-1] = round(sum([mat[i][i] for i in range(num_out)]) / len(y_test), 3)
    except:
        mat[-1][-1] = 0.0

    return mat

def print_confusion_matrix(pred, y_test):
    mat = confusion_matrix(pred, y_test)
    ln = max(len(str(len(y_test))), 5)

    for l in mat:
        print("[" + ', '.join([f"{x:<{ln}}" for x in l]) + "]")