import numpy as np

from utils import onehot, softmax

class RedeNeural:
    def __init__(self, tamanhos: tuple, taxa_aprendizado=0.025, momento=0.75, seed=42):
        rng = np.random.default_rng(seed)
        self.ta = taxa_aprendizado
        self.momento = momento
        self.tamanhos = tamanhos # tamanho de cada camada. isso é algo como {784, 128, 64, 10} para o q fiz no MNIST

        # inicializacao de Kaiming. Talvez seja legal separar inicializações como funções próprias? Essa é uma função de inicialização pra trabalhar com ReLU
        self.pesos = [
            (rng.standard_normal((num_entradas, num_saidas)) * np.sqrt(2/num_entradas)).astype(np.float32)
            for num_entradas, num_saidas in zip(tamanhos[:-1], tamanhos[1:])] # talvez seja uma boa explicar? na prática, essa linha faz duplas de entrada e saida

        self.bias = [np.zeros((1, num_saidas), dtype=np.float32) for num_saidas in tamanhos[1:]]
        self.vpeso = [np.zeros_like(pesos) for pesos in self.pesos]
        self.vbias = [np.zeros_like(bias) for bias in self.bias]

    def forward(self, X):
        ativacoes = [X]
        a = X
        ultima = len(self.pesos) - 1
        for i, (peso, bias) in enumerate(zip(self.pesos, self.bias)):
            z = a @ peso + bias
            a = softmax(z) if i == ultima else np.maximum(0, z)
            ativacoes.append(a)
        return ativacoes

    def backward(self, ativacoes, y_onehot):
        n = y_onehot.shape[0]
        dz = (ativacoes[-1] - y_onehot) / n
        for i in reversed(range(len(self.pesos))):
            dpesos = ativacoes[i].T @ dz
            dbias = dz.sum(axis=0, keepdims=True)
            if i > 0: dz = (dz @ self.pesos[i].T) * (ativacoes[i] > 0)
            self.vpeso[i] = self.momento * self.vpeso[i] - self.ta * dpesos
            self.vbias[i] = self.momento * self.vbias[i] - self.ta * dbias
            self.pesos[i] += self.vpeso[i]
            self.bias[i] += self.vbias[i]

    def prever(self, X):
        return self.forward(X)[-1].argmax(axis=1)

    def treinar(self, X, y, X_val, y_val, epocas=20, batch=64, seed=42):
        rng = np.random.default_rng(seed)
        Y = onehot(y)
        n = X.shape[0]
        for epoca in range(1, epocas+1):
            idx = rng.permutation(n)
            perda_total = 0.0
            for ini in range(0, n, batch):
                sel = idx[ini:ini + batch]
                ativ = self.forward(X[sel])
                probabilidades = ativ[-1]
                perda_total += -np.sum(Y[sel]*np.log(probabilidades + 1e-9))
                self.backward(ativ, Y[sel])
            acc_tr = (self.prever(X) == y).mean()
            acc_val = (self.prever(X_val) == y_val).mean()
            print(f"Época {epoca:2d}/{epocas} | perda: {perda_total / n:.4f} "
                  f"| acc treino: {acc_tr:.4f} | acc teste: {acc_val:.4f}")
