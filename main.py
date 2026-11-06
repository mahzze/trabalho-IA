import sys
from utils import carregar_csv
from rede import RedeNeural
from metricas import avaliar

def main():
    caminho_treino = sys.argv[1] if len(sys.argv) > 1 else "mnist_train.csv"
    caminho_teste = sys.argv[2] if len(sys.argv) > 2 else "mnist_test.csv"

    ep = int(input("Quantas épocas?  "))
    n = int(input("Quantas camadas?  "))
    camadas_arr = []
    for i in range(n):
        camadas_arr.append(int(input(f"Tamanho da camada {i+1}")))
    camadas = tuple(camadas_arr)

    print("Carregando dados...")
    X_tr, y_tr = carregar_csv(caminho_treino)
    X_te, y_te = carregar_csv(caminho_teste)
    print(f"Treino: {X_tr.shape} | Teste: {X_te.shape}")

    rede = RedeNeural(tamanhos=camadas, taxa_aprendizado=0.05, momento=0.9)
    rede.treinar(X_tr, y_tr, X_te, y_te, epocas=ep, batch=64)

    y_pred = rede.prever(X_te)
    acc = (y_pred == y_te).mean()
    print(f"\nAcurácia final no teste: {acc * 100:.2f}%")

    avaliar(y_te, y_pred)

if __name__ == "__main__":
    main()
