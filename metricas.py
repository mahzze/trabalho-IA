#ESTE ARQUIVO FOI GERADO POR IA.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

def _divisao_segura(num, den):
    """Divide evitando divisão por zero (resultado 0 quando den == 0)."""
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den != 0)


def avaliar(y_real, y_pred, arquivo_imagem="matriz_confusao.png"):
    classes = np.arange(10)
    cm = confusion_matrix(y_real, y_pred, labels=classes)

    # Valores por classe (abordagem "um contra todos") extraídos da matriz
    tp = np.diag(cm)                    # acertos da classe
    fp = cm.sum(axis=0) - tp            # previstos como a classe, mas eram outra
    fn = cm.sum(axis=1) - tp            # eram a classe, mas previstos como outra
    tn = cm.sum() - (tp + fp + fn)      # todo o resto

    precisao = _divisao_segura(tp, tp + fp)
    revocacao = _divisao_segura(tp, tp + fn)          # sensibilidade / recall
    especificidade = _divisao_segura(tn, tn + fp)
    f1 = _divisao_segura(2 * precisao * revocacao, precisao + revocacao)
    acuracia = tp.sum() / cm.sum()

    tabela = pd.DataFrame({
        "Precisão": precisao,
        "Revocação": revocacao,
        "Especificidade": especificidade,
        "F1": f1,
        "Suporte": cm.sum(axis=1),
    }, index=[f"Dígito {c}" for c in classes])

    print("\n=== Matriz de confusão (linhas = real, colunas = previsto) ===")
    print(pd.DataFrame(cm, index=classes, columns=classes))

    print("\n=== Métricas por classe (calculadas a partir da matriz) ===")
    print(tabela.round(4).to_string())

    print("\n=== Médias ===")
    print(f"Acurácia global:        {acuracia:.4f}")
    print(f"Macro (média simples):  precisão={precisao.mean():.4f} | "
          f"revocação={revocacao.mean():.4f} | F1={f1.mean():.4f}")
    pesos = cm.sum(axis=1) / cm.sum()
    print(f"Ponderada (por suporte): precisão={np.sum(precisao * pesos):.4f} | "
          f"revocação={np.sum(revocacao * pesos):.4f} | F1={np.sum(f1 * pesos):.4f}")

    # Conferência com o relatório pronto do scikit-learn
    print("\n=== classification_report (scikit-learn) ===")
    print(classification_report(y_real, y_pred, digits=4))

    # Gráfico: contagens e versão normalizada por linha (revocação)
    fig, eixos = plt.subplots(1, 2, figsize=(16, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_real, y_pred, labels=classes, cmap="Blues",
        values_format="d", ax=eixos[0], colorbar=False)
    eixos[0].set_title("Matriz de confusão (contagens)")
    ConfusionMatrixDisplay.from_predictions(
        y_real, y_pred, labels=classes, cmap="Blues", normalize="true",
        values_format=".2f", ax=eixos[1], colorbar=False)
    eixos[1].set_title("Matriz de confusão (normalizada por linha)")
    for ax in eixos:
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
    plt.tight_layout()
    plt.savefig(arquivo_imagem, dpi=150)
    print(f"\nImagem salva em: {arquivo_imagem}")
    plt.show()
