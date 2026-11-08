# =============================================================
# vns.py — Heurística construtiva + VNS (Variable Neighborhood Search)
# =============================================================

import numpy as np
from avaliacao import e_viavel
from dados import (
    N_MINERIOS, N_PILHAS, elegivel, dmin, dmax, n_cam_alvo, preco
)


def solucao_inicial(seed=None):
    rng = np.random.default_rng(seed)
    X = np.zeros((N_MINERIOS, N_PILHAS), dtype=int)

    # Passo 1: alocar mínimos obrigatórios (dmin > 0)
    for i in range(N_MINERIOS):
        if dmin[i] == 0:
            continue
        pilhas_eleg = np.array([p for p in range(N_PILHAS) if elegivel[i, p]])
        if len(pilhas_eleg) == 0:
            continue
        rng.shuffle(pilhas_eleg)
        restante = int(dmin[i])
        for p in pilhas_eleg:
            if restante <= 0:
                break
            X[i, p] += 1
            restante -= 1
        idx = 0
        while restante > 0:
            p = pilhas_eleg[idx % len(pilhas_eleg)]
            X[i, p] += 1
            restante -= 1
            idx += 1

    # Passo 2: completar cada pilha com minérios mais baratos elegíveis
    ordem_preco = np.argsort(preco)
    for p in range(N_PILHAS):
        faltam = int(n_cam_alvo[p] - X[:, p].sum())
        if faltam <= 0:
            continue
        for i in ordem_preco:
            if faltam <= 0:
                break
            if not elegivel[i, p]:
                continue
            uso_atual = X[i, :].sum()
            pode = min(faltam, int(dmax[i] - uso_atual))
            if pode > 0:
                X[i, p] += pode
                faltam -= pode

        # Mantém uma solução candidata mesmo quando os limites não permitem
        # completar a pilha; a inviabilidade será medida pela função de avaliação.
        if faltam > 0:
            candidatos = [i for i in ordem_preco if elegivel[i, p]]
            if candidatos:
                X[candidatos[0], p] += faltam
    return X


def vizinhanca_N1(X, rng):
    """Substitui um caminhão de um minério presente por outro elegível na mesma pilha."""
    Xn = X.copy()
    p = int(rng.integers(0, N_PILHAS))
    presentes = np.flatnonzero(Xn[:, p] > 0)
    if len(presentes) == 0:
        return Xn
    i_sai = int(rng.choice(presentes))
    candidatos = np.flatnonzero(elegivel[:, p])
    candidatos = candidatos[candidatos != i_sai]
    if len(candidatos) == 0:
        return Xn
    j_entra = int(rng.choice(candidatos))
    Xn[i_sai, p] -= 1
    Xn[j_entra, p] += 1
    return Xn


def vizinhanca_N2(X, rng):
    """Troca um caminhão entre duas pilhas, respeitando a elegibilidade de destino."""
    Xn = X.copy()
    p1, p2 = rng.choice(N_PILHAS, size=2, replace=False)
    presentes_p1 = np.flatnonzero(Xn[:, p1] > 0)
    presentes_p2 = np.flatnonzero(Xn[:, p2] > 0)
    if len(presentes_p1) == 0 or len(presentes_p2) == 0:
        return Xn
    i = int(rng.choice(presentes_p1))
    j = int(rng.choice(presentes_p2))
    if not (elegivel[i, p2] and elegivel[j, p1]):
        return Xn
    Xn[i, p1] -= 1
    Xn[j, p2] -= 1
    Xn[i, p2] += 1
    Xn[j, p1] += 1
    return Xn


def vizinhanca_N3(X, rng):
    """
    Escolhe uma pilha, identifica o minério mais caro presente e substitui
    todos os seus caminhões pelo minério elegível mais barato para essa pilha.
    """
    Xn = X.copy()
    p = int(rng.integers(0, N_PILHAS))
    presentes = np.flatnonzero(Xn[:, p] > 0)
    if len(presentes) == 0:
        return Xn
    i_sai = int(max(presentes, key=lambda i: preco[i]))
    candidatos = [j for j in range(N_MINERIOS) if elegivel[j, p] and j != i_sai]
    if not candidatos:
        return Xn
    j_entra = int(min(candidatos, key=lambda j: preco[j]))
    qtd = int(Xn[i_sai, p])
    Xn[i_sai, p] = 0
    Xn[j_entra, p] += qtd
    return Xn


def perturbacao(X, k, rng):
    """Aplica k movimentos, sorteando N1, N2 ou N3 a cada passo."""
    Xp = X.copy()
    vizinhos = (vizinhanca_N1, vizinhanca_N2, vizinhanca_N3)
    for _ in range(k):
        viz = rng.choice(vizinhos)
        Xp = viz(Xp, rng)
    return Xp


def busca_local(X, obj_func, max_iter, rng, max_tentativas_sem_melhoria=30,
                vizinhancas=None):
    """
    Busca local First Improvement por amostragem aleatória.
    Avalia no máximo max_iter movimentos; aceita o primeiro que melhora a
    solução e encerra após max_tentativas_sem_melhoria movimentos consecutivos
    sem melhora. A parada não certifica ótimo local global.
    """
    if vizinhancas is None:
        vizinhancas = [vizinhanca_N1, vizinhanca_N2, vizinhanca_N3]

    X_atual = X.copy()
    f_atual = obj_func(X_atual)
    tentativas_sem_melhoria = 0

    for _ in range(max_iter):
        if tentativas_sem_melhoria >= max_tentativas_sem_melhoria:
            break
        viz = vizinhancas[int(rng.integers(0, len(vizinhancas)))]
        Xn = viz(X_atual, rng)
        fn = obj_func(Xn)
        if fn < f_atual:
            X_atual, f_atual = Xn, fn
            tentativas_sem_melhoria = 0
        else:
            tentativas_sem_melhoria += 1

    return X_atual, f_atual


def vns(obj_func, seed=None, max_iter=1500, k_max=3, iter_bl=150,
        max_tentativas_sem_melhoria=30, viavel_func=e_viavel):
    """
    Executa VNS e preserva a melhor solução viável encontrada, se houver.
    viavel_func permite acrescentar restrições (ex.: ε-restrito) à checagem
    de viabilidade; por padrão usa apenas as restrições físicas do problema.
    """
    e_viavel = viavel_func
    rng = np.random.default_rng(seed)
    X_atual = solucao_inicial(seed=seed)
    f_atual = obj_func(X_atual)

    X_melhor_viavel = X_atual.copy() if e_viavel(X_atual) else None
    f_melhor_viavel = obj_func(X_melhor_viavel) if X_melhor_viavel is not None else float("inf")

    X_melhor_qualquer = X_atual.copy()
    f_melhor_qualquer = f_atual
    historico = [f_melhor_viavel if np.isfinite(f_melhor_viavel) else f_atual]
    k = 1

    for _ in range(max_iter):
        Xp = perturbacao(X_atual, k, rng)
        Xp_bl, fp_bl = busca_local(
            Xp, obj_func, iter_bl, rng,
            max_tentativas_sem_melhoria=max_tentativas_sem_melhoria
        )

        # Registra candidatos independentemente do critério de aceitação do VNS.
        for candidato, valor in ((Xp, obj_func(Xp)), (Xp_bl, fp_bl)):
            if valor < f_melhor_qualquer:
                X_melhor_qualquer = candidato.copy()
                f_melhor_qualquer = valor
            if e_viavel(candidato):
                valor_viavel = obj_func(candidato)
                if valor_viavel < f_melhor_viavel:
                    X_melhor_viavel = candidato.copy()
                    f_melhor_viavel = valor_viavel

        if fp_bl < f_atual:
            X_atual, f_atual = Xp_bl.copy(), fp_bl
            k = 1
        else:
            k = 1 if k >= k_max else k + 1

        if e_viavel(X_atual):
            valor_atual = obj_func(X_atual)
            if valor_atual < f_melhor_viavel:
                X_melhor_viavel = X_atual.copy()
                f_melhor_viavel = valor_atual

        historico.append(
            f_melhor_viavel if np.isfinite(f_melhor_viavel) else f_melhor_qualquer
        )

    # Se nenhuma solução viável foi encontrada, devolve a melhor candidata
    # penalizada para que main_monoobj.py possa relatar suas violações, sem a rotular viável.
    if X_melhor_viavel is not None:
        return X_melhor_viavel, f_melhor_viavel, historico
    return X_melhor_qualquer, f_melhor_qualquer, historico
