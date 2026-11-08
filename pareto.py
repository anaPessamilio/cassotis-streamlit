# =============================================================
# pareto.py — Identificação de soluções não dominadas (minimização)
# =============================================================

import numpy as np


def nao_dominados(F):
    """
    F: array (n, m) com os objetivos (todos a minimizar).
    Retorna máscara booleana das linhas não dominadas.
    Duplicatas exatas são mantidas apenas uma vez.
    """
    F = np.asarray(F, dtype=float)
    n = len(F)
    mantem = np.ones(n, dtype=bool)
    for i in range(n):
        if not mantem[i]:
            continue
        domina_i = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if domina_i.any():
            mantem[i] = False
            continue
        iguais = np.all(F == F[i], axis=1)
        iguais[: i + 1] = False
        mantem[iguais] = False
    return mantem
