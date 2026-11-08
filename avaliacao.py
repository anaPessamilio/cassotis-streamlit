# =============================================================
# avaliacao.py — Funções objetivo e verificação de restrições
# =============================================================

import numpy as np
from dados import (
    preco, sio2_m, al2o3_m, dmin, dmax,
    elegivel, n_cam_alvo, alvo_sio2, alvo_al2o3,
    min_sio2, max_sio2, min_al2o3, max_al2o3
)

def teor_pilha(X, indicador):
    numerador = indicador @ X
    denominador = X.sum(axis=0)
    denom_safe = np.where(denominador == 0, 1, denominador)
    return numerador / denom_safe

def f1(X):
    """Custo total. Unidade: R$. 2kt = 2000 t por camião."""
    return float(np.sum(X * preco[:, np.newaxis]) * 2000)

def f2(X):
    q_sio2 = teor_pilha(X, sio2_m)
    return float(np.sum((q_sio2 - alvo_sio2) ** 2))

def f3(X):
    q_al2o3 = teor_pilha(X, al2o3_m)
    return float(np.sum((q_al2o3 - alvo_al2o3) ** 2))

def violar_massa(X):
    total_por_pilha = X.sum(axis=0)
    return int(np.sum(np.abs(total_por_pilha - n_cam_alvo)))

def violar_qualidade(X):
    q_sio2  = teor_pilha(X, sio2_m)
    q_al2o3 = teor_pilha(X, al2o3_m)
    viol  = np.sum(np.maximum(0, min_sio2  - q_sio2))
    viol += np.sum(np.maximum(0, q_sio2  - max_sio2))
    viol += np.sum(np.maximum(0, min_al2o3 - q_al2o3))
    viol += np.sum(np.maximum(0, q_al2o3 - max_al2o3))
    return float(viol)

def violar_elegibilidade(X):
    return int(np.sum(X[~elegivel]))

def violar_disponibilidade(X):
    uso = X.sum(axis=1)
    viol  = np.sum(np.maximum(0, dmin - uso))
    viol += np.sum(np.maximum(0, uso  - dmax))
    return int(viol)

def e_viavel(X):
    return (
        violar_massa(X) == 0 and
        violar_qualidade(X) == 0.0 and
        violar_elegibilidade(X) == 0 and
        violar_disponibilidade(X) == 0
    )

#Modificado para entrega 2 (correção entrega 1)
def penalidade(X, scale=1.0):
    return scale * (
       100.0 * violar_massa(X) +
        50.0  * violar_qualidade(X) +
        100.0 * violar_elegibilidade(X) +
        50.0  * violar_disponibilidade(X)
    )

def f1_penalizado(X):
    # f1 está na casa de 1e8, multiplicamos a escala da penalidade por 1e6
    return f1(X) + penalidade(X, scale=1e6)

def f2_penalizado(X):
    return f2(X) + penalidade(X, scale=1.0)

def f3_penalizado(X):
    return f3(X) + penalidade(X, scale=1.0)

def relatorio_solucao(X, verbose=True):
    from dados import PILHAS

    custo  = f1(X)
    dev_s  = f2(X)
    dev_a  = f3(X)
    viavel = e_viavel(X)

    q_sio2  = teor_pilha(X, sio2_m)
    q_al2o3 = teor_pilha(X, al2o3_m)

    if verbose:
        print(f"{'='*55}")
        print(f"  Custo total : R$ {custo:,.2f}")
        print(f"  f2 (ΣΔSiO2²): {dev_s:.6f}")
        print(f"  f3 (ΣΔAl2O3²): {dev_a:.6f}")
        print(f"  Viável      : {viavel}")
        print(f"  Viol. massa : {violar_massa(X)}")
        print(f"  Viol. qual  : {violar_qualidade(X):.4f}")
        print(f"  Viol. eleg  : {violar_elegibilidade(X)}")
        print(f"  Viol. disp  : {violar_disponibilidade(X)}")
        print(f"{'-'*55}")
        print(f"  {'Pilha':<10} {'Sinter':>6} {'nCam':>5} {'SiO2%':>7} {'Al2O3%':>8}")
        print(f"{'-'*55}")
        for p, pilha in enumerate(PILHAS):
            s = pilha["sinter"]
            print(f"  {pilha['nome']:<10} {s:>6} {X[:,p].sum():>5} "
                  f"{q_sio2[p]:>7.3f} {q_al2o3[p]:>8.3f}")
        print(f"{'='*55}")
    return {"f1": custo, "f2": dev_s, "f3": dev_a, "viavel": viavel}

#Modificado para entrega 3
def f4_homogeneidade(X):
    """
    f4: Mede a heterogeneidade das receitas entre pilhas do mesmo sínter.
    Calcula a soma dos desvios quadráticos das alocações de caminhões de cada
    pilha em relação à receita média das pilhas do seu respectivo equipamento.
    """
    idx_s1 = list(range(0, 5))  # Pilhas 1 a 5 (Sínter 1)
    idx_s2 = list(range(5, 10)) # Pilhas 6 a 10 (Sínter 2)
    
    media_s1 = X[:, idx_s1].mean(axis=1, keepdims=True)
    media_s2 = X[:, idx_s2].mean(axis=1, keepdims=True)
    
    dev_s1 = np.sum((X[:, idx_s1] - media_s1) ** 2)
    dev_s2 = np.sum((X[:, idx_s2] - media_s2) ** 2)
    return float(dev_s1 + dev_s2)