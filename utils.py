import numpy as np
import pandas as pd

# no fim, fiz as da sigmoide e derivada pra nada kskksksksksksks
def sigmoide(x: float):
    return 1/(1+np.exp(-x))

def sigmoide_derivada(x: float):
    return sigmoide(x)*(1-sigmoide(x))

def softmax(x: np.float32):
    x = x - x.max(axis=1, keepdims=True)
    return np.exp(x)/np.exp(x).sum(axis=1, keepdims=True)

def carregar_csv(path):
    # Os dados são valores inteiros de 0 a 255, equivalendo aos valores de cada pixel
    df = pd.read_csv(path)
    dados = df.to_numpy(dtype=np.float32)
    y = dados[:, 0].astype(np.int8) # labels de 0 a 9
    X = dados[:, 1:] / 255.0
    return X, y

def onehot(y, num_classes=10):
    """
    num_classes vai ser 10 para o MNIST, e (talvez) um outro valor para outras
    bases, entao vou deixar como parametro.
    """
    saida = np.zeros((y.size, num_classes), dtype=np.float32)
    saida[np.arange(y.size), y] = 1.0
    return saida
