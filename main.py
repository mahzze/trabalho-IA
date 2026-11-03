import sys
from utils import carregar_csv
from rede import RedeNeural

def main():
    caminho_treino = sys.argv[1] if len(sys.argv) > 1 else "mnist_train.csv"
    caminho_teste = sys.argv[2] if len(sys.argv) > 2 else "mnist_test.csv"

    print("Carregando dados...")
    X_tr, y_tr = carregar_csv(caminho_treino)
    X_te, y_te = carregar_csv(caminho_teste)
    print(f"Treino: {X_tr.shape} | Teste: {X_te.shape}")

    rede = RedeNeural(tamanhos=(784, 128, 64, 10), taxa_aprendizado=0.05, momento=0.9)
    rede.treinar(X_tr, y_tr, X_te, y_te, epocas=15, batch=64)

    y_pred = rede.prever(X_te)
    acc = (y_pred == y_te).mean()
    print(f"\nAcurácia final no teste: {acc * 100:.2f}%")

    avaliar(y_te, y_pred)

if __name__ == "__main__":
    main()
