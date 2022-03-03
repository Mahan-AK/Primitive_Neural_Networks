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
    if s == "1": return [1, -1, -1].copy()
    elif s == "2": return [-1, 1, -1].copy()
    elif s == "3": return [-1, -1, 1].copy()

with open("ann-train.data", 'r') as f:
    #f.readline()
    for line in f.readlines():
        t = line.strip('\n').split(' ')
        X.append([float(t[0])] + list(map(int, t[1:16])) + list(map(float, t[16:-3])))
        y.append(encode(t[-3]))

X, y = shuffle(X, y)

X_train, X_valid = X[:int(len(X)*(8/10))], X[int(len(X)*(8/10)):]
y_train, y_valid = y[:int(len(y)*(8/10))], y[int(len(y)*(8/10)):]

# using bipolar sigmoid function
def act(x):
    return math.tanh(x/2)

# derivitive of f:
def der_act(x):
    return 1/(math.cosh(x) + 1)

# w_h, b_h, w_o, b_o = train_TLP(X_train, y_train, X_valid, y_valid, hidden_layer_num=4, alpha=0.033, f=act, df=der_act)

w_h = [[-351.1770063938664, -40.323020445259054, -132.8439750904889, 5.281200532683861], [-16.05430715522109, 281.0386083626906, -140.9624573509464, -3.8726585199595513], [94.81020463755833, -50.01437530236639, 272.05939230614547, -1.6061945924718597], [51.986502876933365, -117.35001709037829, -76.74719652095737, -0.20287389697928712], [-0.29941531250961745, -0.10942527648849429, -0.4371493460408298, -0.6021693092280351], [0.022094704294219128, -0.11816962736568047, 0.10580780401885638, 0.06545405297553963], [-0.20050326699029472, -0.46404188032665517, 0.3062663372025694, -0.49851411138720864], [0.41908370171504145, 0.24795146914914024, 0.4903322615172288, -0.1059138962660475], [0.19885026543875678, 0.3246382221906156, -0.17428355082639174, 0.5692232365784073], [-0.5062101214081379, 0.23924192901453062, -0.12725141925098754, 0.213170528882803], [-0.2500292160406001, -0.5189742293515565, -0.421302314978043, 0.04239857931267873], [-0.09136647415107513, -0.18620930646173078, 0.36347897971021725, -0.12191465774070043], [0.13455558260998657, 0.34221303355196636, -0.09305681041868251, -0.1550832317342224], [0.32942063999562127, 0.3076629041866297, 0.210283409613063, -0.2907076088301711], [-0.03798741062704603, -0.3399329496749143, -0.04587387891344308, 0.025981865815408116], [-0.4752849264987497, -0.5408307303418891, 0.25663364491168184, -0.1397025390635325], [0.3444311052261186, 0.18402347706041364, -0.38408486889869686, 0.233787140790531], [-0.362888130983958, -0.15781679830559273, -0.34703769128399026, 0.32378079488040856], [-0.13439432796399495, -0.2505335609669981, -0.23660268902422235, -0.5765187876860256], [-0.7816255208838063, -0.04532554038526936, 0.197822021207603, -0.09221021357176153], [-0.7983942633779882, -0.045635850335594745, -0.30758757832700323, -0.036850748577711014]]
w_o = [[0.37032623122217434, 0.061850323485796666, -0.4602953095378637], [-0.1907479248908479, -0.4602198667737444, -0.32812052313230367], [0.35960860371752557, -0.36341820415378656, -0.3193983727824731], [0.3679097008704486, -0.46346391817862376, -0.46295403126898216]]
b_h = [119.56459689673369, -16.635136812401605, 284.6215881036008, -2.048956363942426]
b_o = [-0.39233464333205315, 0.32884019278111287, -0.21067249506829044]

print("Hidden Layer:")
print(f"Weights: {w_h}")
print(f"Biases: {b_h}")
print()

print("Output Layer:")
print(f"Weights: {w_o}")
print(f"Biases: {b_o}")
print()

print(len(w_h[0]), len(b_h))
print(len(w_o[0]), len(b_o))

X_test = []
y_test = []

with open("ann-test.data", 'r') as f:
    #f.readline()
    for line in f.readlines():
        t = line.strip('\n').split(' ')
        X_test.append([float(t[0])] + list(map(int, t[1:16])) + list(map(float, t[16:-3])))
        y_test.append(encode(t[-3]))

pred = predict(X_test, w_h, b_h, w_o, b_o)

t = sum([xs == ys for xs, ys in zip(pred, y_test)])

print(f"Accuracy (tested on {len(y_test)} samples): {t/len(X_test) * 100:.3f}%")

print("\nConfusion matrix:")
print_confusion_matrix(pred, y_test)