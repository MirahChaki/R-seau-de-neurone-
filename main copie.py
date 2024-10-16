# pour importer numpy : file -> New projects setup
# -> Preference for new projects-> Python interpreter ->
# select python interpreter -> + -> numpy -> install package -> croix

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics import log_loss

# genere un data set contenant 100 lignes et 2 variables
# grace à la fonction make blobs que l'on trouve dans sklearn
X,y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
X = X.T
y = y.reshape((1, y.shape[0]))

print("dimension de X ", X.shape)
print('dimension de y', y.shape)

plt.scatter(X[0,:],X[1,:], c=y, cmap='summer')
plt.show()

def initialisation(n0,n1,n2):
    W1=np.random.randn(n1,n0)
    b1=np.random.randn(n1,1)
    W2=np.random.randn(n2,n1)
    b2=np.random.randn(n2,1)

    # On met les parametres dans un dictionnaire, qu'on retourne
    #entre crochet ces la clés correspondante
    parametres = {
        'W1':W1,
        'b1':b1,
        'W2':W2,
        'b2':b2
    }
    return (parametres)

def forward_propagation(X,parametres):
    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']

    Z1 = W1.dot(X) + b1
    A1 = 1 / (1 + np.exp(-Z1))
    Z2 = W2.dot(X) + b2
    A2 = 1 / (1 + np.exp(-Z2))

    activations ={
        'A1' : A1,
        'A2' : A2
    }
    return activations

def back_propagation(X,y, activations, parametres):
    A1 = activations['A1']
    A2 = activations['A2']
    W2 = parametres['W2']

    # longueur de y
    m = y.shape[1]

    dZ2 = A2 - y
    dW2 = 1 / m * dZ2.dot(A1.T) # transposé
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True) # drodcasting

    dZ1 = np.dot(W2.T,dZ2) * A1 * (1 - A1)
    dW1 = 1 / m * dZ1.dot(X.T)  # transposé
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {
        'dW1' : dW1,
        'db1' : db1,
        'dW2' : dW2,
        'db2' : db2
    }
    return gradients

def update(gradients, parametres, learning_rate):

    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']

    dW1 = gradients['dW1']
    db1 = gradients['db1']
    dW2 = gradients['dW2']
    db2 = gradients['db2']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parametres = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }
    return parametres

def predict(X, parametres):
    activations = forward_propagation(X, parametres)
    A2 = activations['A2']
    # print(A) probabilite
    return A2 >= 0.5

from sklearn.metrics import accuracy_score

#le nombre de couche souhaité
def neural_network(X_train, y_train, n1, learning_rate = 0.1, n_iter = 1000):

    # initialisation W, b
    n0 = X_train.shape[0]
    n2 = y_train.shape[0]
    parametres = initialisation(n0,n1,n2)

    train_loss = []
    train_acc = []

    for i in range(n_iter):
        activations = forward_propagation(X_train, parametres)
        gradients = back_propagation(X_train, y_train, activations, parametres)
        parametres = update(gradients, parametres, learning_rate)

        # Le reste est pour visualiser les courbes d'apprentissage
        # toutes les 10 iterations on ajoute le loss dans une liste vide
        #en comprant les donnée y du y_train avec les activations
        if i %10 == 0 :
            train_loss.append(log_loss(y_train, activations['A2']))
            y_pred = predict (X_train, parametres)
            current_accuracy = accuracy_score(y_train.flatten(), y_pred.flatten())#applatire les tableaux
            train_acc.append(current_accuracy)

    plt.figure(figsize=(14,4))

    plt.subplot(1,2,1)
    plt.plot(train_loss, label='train loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_acc, label='train acc')
    plt.legend()
    plt.show()


    return parametres

#train_loss : marge apprentissage
#train_acc : exactitude
parametres = neural_network(X,y, n1=2, n_iter=10000, learning_rate=0.1)