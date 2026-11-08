# =============================================================
# mcda.py — Métodos MCDA (AHP e PROMETHEE II) Unificados
# =============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import avaliacao
from pareto import nao_dominados
from dados import (
    PILHAS, NOMES_MINERIOS, N_MINERIOS, N_PILHAS,
    sio2_m, al2o3_m, alvo_sio2, alvo_al2o3,
    min_sio2, max_sio2, min_al2o3, max_al2o3
)


# f4 tem uma única definição, em avaliacao.py (antes havia uma cópia aqui).
f4_homogeneidade = avaliacao.f4_homogeneidade


def carregar_fronteira_global_unificada(arq_sp='solucoes_soma_ponderada.npz', 
                                       arq_eps='solucoes_epsilon_restrito.npz'):
    """
    Carrega as soluções não dominadas da Soma Ponderada e do ε-Restrito,
    concatena os dois conjuntos e re-filtra a não dominância cruzada entre elas.
    Retorna as matrizes X unificadas e a matriz de decisão com os 4 critérios.
    """
    sol_sp = np.load(arq_sp)
    sol_eps = np.load(arq_eps)
    
    X_sp = sol_sp['X'][sol_sp['nao_dominada']]
    X_eps = sol_eps['X'][sol_eps['nao_dominada']]
    
    X_uniao = np.concatenate([X_sp, X_eps], axis=0)
    
    matriz_uniao = np.array([
        [avaliacao.f1(X), avaliacao.f2(X), avaliacao.f3(X), f4_homogeneidade(X)]
        for X in X_uniao
    ])
    
    # Filtro de Pareto considerando f1, f2 e f3
    mask_global = nao_dominados(matriz_uniao[:, :3])
    
    X_pareto_global = X_uniao[mask_global]
    matriz_pareto_global = matriz_uniao[mask_global]
    
    return X_pareto_global, matriz_pareto_global


def carregar_referencias(arq="referencias.json"):
    """Valores isolados da Entrega 1 (mín e máx por objetivo) — referência de normalização do enunciado."""
    if not os.path.exists(arq):
        return None
    with open(arq, "r", encoding="utf-8") as f:
        r = json.load(f)
    lo = np.array([r["f1"]["min"], r["f2"]["min"], r["f3"]["min"]], dtype=float)
    hi = np.array([r["f1"]["max"], r["f2"]["max"], r["f3"]["max"]], dtype=float)
    return lo, hi


def faixa_normalizacao(matriz_decisao, matriz_pareto, referencias=None):
    """
    Faixa [lo, hi] por critério, usada na normalização do AHP.
    f1..f3: referência da Entrega 1 (se houver); f4 não tem referência na Entrega 1,
    então usa o mín/máx da fronteira global. Sem referências, usa a fronteira para os 4.
    """
    lo = matriz_pareto.min(axis=0).astype(float)
    hi = matriz_pareto.max(axis=0).astype(float)
    if referencias is not None:
        lo[:3], hi[:3] = referencias[0], referencias[1]
    return lo, hi


def executar_ahp(matriz_decisao, saaty_criterios, normalizacao="inverso", faixa=None):
    """
    AHP para minimização.
    matriz_decisao: (N_alt, 4) — colunas: [f1, f2, f3, f4]
    saaty_criterios: (4, 4) — Matriz de comparação par a par de Saaty
    normalizacao:
      "inverso"  — v_ij = (1/x_ij) / soma_i(1/x_ij)   (versão original; MUITO sensível à escala: um critério
                   que quase não varia entre as alternativas, como o custo, perde quase todo o peso)
      "faixa"    — v_ij = (hi_j - x_ij) / (hi_j - lo_j), com faixa=(lo, hi) por critério
                   (min-max linear; preserva o peso declarado na matriz de Saaty)
    """
    evals, evecs = np.linalg.eig(saaty_criterios)
    max_idx = np.argmax(np.real(evals))
    lambda_max = np.real(evals[max_idx])
    pesos = np.real(evecs[:, max_idx])
    pesos = pesos / np.sum(pesos)
    
    n = 4
    ci = (lambda_max - n) / (n - 1)
    cr = ci / 0.90
    
    if normalizacao == "faixa":
        lo, hi = faixa
        norm_decisao = (hi - matriz_decisao) / (hi - lo)
    else:
        inv_decisao = 1.0 / matriz_decisao
        norm_decisao = inv_decisao / inv_decisao.sum(axis=0, keepdims=True)
    
    scores = norm_decisao @ pesos
    return scores, pesos, cr


def executar_promethee_ii(matriz_decisao, pesos, p_limiares, q_limiares):
    """
    PROMETHEE II com Função Tipo V (Linear com Indiferença) para minimização.
    """
    n_alt, n_crit = matriz_decisao.shape
    pi_matrix = np.zeros((n_alt, n_alt))
    
    for i in range(n_alt):
        for j in range(n_alt):
            if i == j:
                continue
            pref_sum = 0.0
            for k in range(n_crit):
                d = matriz_decisao[j, k] - matriz_decisao[i, k]
                if d <= q_limiares[k]:
                    P = 0.0
                elif d >= p_limiares[k]:
                    P = 1.0
                else:
                    P = (d - q_limiares[k]) / (p_limiares[k] - q_limiares[k])
                pref_sum += pesos[k] * P
            pi_matrix[i, j] = pref_sum
            
    phi_plus = pi_matrix.sum(axis=1) / (n_alt - 1)
    phi_minus = pi_matrix.sum(axis=0) / (n_alt - 1)
    phi_net = phi_plus - phi_minus
    return phi_net, phi_plus, phi_minus


def plot_projecoes_pareto(matriz_pareto_global, matriz_decisao, idx_vencedor_ahp, idx_vencedor_promethee, filename="projecoes_pareto_tomada_decisao.png"):
    """
    Gera a figura da letra (e): Projeções bidimensionais f1xf2, f1xf3 e f2xf3
    destacando as soluções da amostra e as escolhas do AHP e PROMETHEE.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Projeções da Fronteira de Pareto e Destaque das Alternativas Escolhidas", fontsize=14, fontweight='bold')

    f1_all = matriz_pareto_global[:, 0] / 1e6
    f2_all = matriz_pareto_global[:, 1]
    f3_all = matriz_pareto_global[:, 2]

    f1_sample = matriz_decisao[:, 0] / 1e6
    f2_sample = matriz_decisao[:, 1]
    f3_sample = matriz_decisao[:, 2]

    f1_ahp, f2_ahp, f3_ahp = f1_sample[idx_vencedor_ahp], f2_sample[idx_vencedor_ahp], f3_sample[idx_vencedor_ahp]
    f1_prom, f2_prom, f3_prom = f1_sample[idx_vencedor_promethee], f2_sample[idx_vencedor_promethee], f3_sample[idx_vencedor_promethee]

    pares = [
        (f1_all, f2_all, f1_sample, f2_sample, f1_ahp, f2_ahp, f1_prom, f2_prom, "Custo (R$ milhões)", "Σ desvio² SiO₂", "f1 x f2"),
        (f1_all, f3_all, f1_sample, f3_sample, f1_ahp, f3_ahp, f1_prom, f3_prom, "Custo (R$ milhões)", "Σ desvio² Al₂O₃", "f1 x f3"),
        (f2_all, f3_all, f2_sample, f3_sample, f2_ahp, f3_ahp, f2_prom, f3_prom, "Σ desvio² SiO₂", "Σ desvio² Al₂O₃", "f2 x f3")
    ]

    for ax, (x_all, y_all, x_s, y_s, x_ahp, y_ahp, x_prom, y_prom, xlab, ylab, title) in zip(axes, pares):
        ax.scatter(x_all, y_all, color='gray', alpha=0.35, label='Fronteira Global', s=25)
        ax.scatter(x_s, y_s, color='blue', marker='o', s=55, label=f'Amostra ({len(matriz_decisao)} alts)')
        ax.scatter(x_ahp, y_ahp, color='crimson', marker='*', s=220, label=f'Escolha AHP (Alt {idx_vencedor_ahp+1})', zorder=5)
        ax.scatter(x_prom, y_prom, color='gold', marker='D', s=130, edgecolor='black', label=f'Escolha PROMETHEE (Alt {idx_vencedor_promethee+1})', zorder=6)
        
        ax.set_xlabel(xlab, fontsize=11)
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Figura salva: {filename}")


def plot_caracterizacao_solucao(X, titulo_solucao, filename):
    """
    Gera a figura da letra (f): Caracterização detalhada da solução escolhida
    (composição de massa das pilhas + teores químicos de SiO2 e Al2O3).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Caracterização Detalhada da Solução Final Recomendada: {titulo_solucao}", fontsize=13, fontweight='bold')
    
    # 1. Composição em massa (kt)
    nomes_pilhas = [p["nome"] for p in PILHAS]
    bottom = np.zeros(N_PILHAS)
    cmap = plt.colormaps["tab20"]
    for i in range(N_MINERIOS):
        vals = X[i, :].astype(float) * 2  # 2 kt por caminhão
        if vals.sum() == 0: 
            continue
        axes[0].bar(nomes_pilhas, vals, bottom=bottom, label=NOMES_MINERIOS[i], color=cmap(i / N_MINERIOS))
        bottom += vals
    axes[0].set_title("Composição das Pilhas por Minério (kt)", fontsize=11)
    axes[0].set_ylabel("Massa Total (kt)")
    axes[0].set_xticks(range(N_PILHAS))
    axes[0].set_xticklabels(nomes_pilhas, rotation=35, ha='right')
    axes[0].legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1, 1))
    
    # 2. Teor SiO2 por pilha
    q_sio2 = avaliacao.teor_pilha(X, sio2_m)
    xs = range(N_PILHAS)
    axes[1].bar(xs, q_sio2, color='#4C72B0', alpha=0.85)
    axes[1].plot(xs, alvo_sio2, 'g--', label='Alvo SiO₂', linewidth=2)
    axes[1].plot(xs, min_sio2, 'r:', label='Limites Mín/Máx', linewidth=1.5)
    axes[1].plot(xs, max_sio2, 'r:', linewidth=1.5)
    axes[1].set_xticks(list(xs))
    axes[1].set_xticklabels(nomes_pilhas, rotation=35, ha='right')
    axes[1].set_title("Teor de SiO₂ por Pilha (%)", fontsize=11)
    axes[1].set_ylabel("SiO₂ (%)")
    axes[1].legend(fontsize=8, loc='lower right')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Teor Al2O3 por pilha
    q_al2o3 = avaliacao.teor_pilha(X, al2o3_m)
    axes[2].bar(xs, q_al2o3, color='#DD8452', alpha=0.85)
    axes[2].plot(xs, alvo_al2o3, 'g--', label='Alvo Al₂O₃', linewidth=2)
    axes[2].plot(xs, min_al2o3, 'r:', label='Limites Mín/Máx', linewidth=1.5)
    axes[2].plot(xs, max_al2o3, 'r:', linewidth=1.5)
    axes[2].set_xticks(list(xs))
    axes[2].set_xticklabels(nomes_pilhas, rotation=35, ha='right')
    axes[2].set_title("Teor de Al₂O₃ por Pilha (%)", fontsize=11)
    axes[2].set_ylabel("Al₂O₃ (%)")
    axes[2].legend(fontsize=8, loc='lower right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Figura salva: {filename}")


if __name__ == "__main__":
    # 1. Carregar a Fronteira Global Unificada (Soma Ponderada + ε-Restrito)
    X_pareto, matriz_pareto = carregar_fronteira_global_unificada()
    print(f"Total de soluções na Fronteira Global Unificada: {len(X_pareto)}")
    
# 2. Selecionar as 8 MELHORES alternativas pelo Ponto Utópico (Menor distância ao ideal)
    m_min = matriz_pareto.min(axis=0)
    m_max = matriz_pareto.max(axis=0)
    m_norm = (matriz_pareto - m_min) / (m_max - m_min)
    
    # Distância Euclidiana em relação ao Ponto Ideal Utópico (0, 0, 0, 0)
    dist_utopia = np.sqrt(np.sum(m_norm**2, axis=1))
    
    # Pega os 8 melhores índices da Fronteira de Pareto Global
    indices = np.argsort(dist_utopia)[:8]
    
    alts_X = X_pareto[indices]
    matriz_decisao = matriz_pareto[indices]
    
    # 3. Definir Matriz de Saaty 4x4: [f1, f2, f3, f4]
    saaty = np.array([
        [1.0, 3.0, 3.0, 5.0],
        [1/3, 1.0, 1.0, 3.0],
        [1/3, 1.0, 1.0, 3.0],
        [1/5, 1/3, 1/3, 1.0]
    ])
    
    # AHP com a normalização ORIGINAL (inverso) e com a normalização por FAIXA (referências da Entrega 1)
    scores_inv, pesos_ahp, cr = executar_ahp(matriz_decisao, saaty, normalizacao="inverso")
    referencias = carregar_referencias()
    if referencias is None:
        print("AVISO: referencias.json não encontrado — a faixa de normalização usará a fronteira global.")
    faixa = faixa_normalizacao(matriz_decisao, matriz_pareto, referencias)
    scores_ahp, _, _ = executar_ahp(matriz_decisao, saaty, normalizacao="faixa", faixa=faixa)

    # 4. Definir Limiares do PROMETHEE II (q: indiferença, p: preferência)
    # Limiares PROPORCIONAIS à faixa de referência de cada critério (iguais aos do app: Q_FRAC, P_FRAC)
    # (versão absoluta original, só para sensibilidade: q=[5e5, 0.20, 0.15, 10], p=[2e6, 1.00, 0.80, 50])
    Q_FRAC, P_FRAC = 0.05, 0.25
    q_lim = list(Q_FRAC * (faixa[1] - faixa[0]))
    p_lim = list(P_FRAC * (faixa[1] - faixa[0]))

    phi_net, _, _ = executar_promethee_ii(matriz_decisao, pesos_ahp, p_lim, q_lim)

    # Identificar alternativas vencedoras
    idx_vencedor_ahp = int(np.argmax(scores_ahp))          # AHP com normalização por faixa (recomendada)
    idx_vencedor_ahp_inv = int(np.argmax(scores_inv))      # AHP com normalização pelo inverso (original)
    idx_vencedor_promethee = int(np.argmax(phi_net))

    # 5. Criar e exportar DataFrame para CSV
    df_res = pd.DataFrame(
        matriz_decisao,
        columns=['f1_custo_R$', 'f2_desvio_sio2', 'f3_desvio_al2o3', 'f4_homogeneidade']
    )
    df_res.insert(0, 'alternativa', [f"Alternativa_{i+1}" for i in range(len(indices))])
    df_res['score_ahp'] = scores_ahp
    df_res['rank_ahp'] = (-scores_ahp).argsort().argsort() + 1
    df_res['escolhida_ahp'] = [i == idx_vencedor_ahp for i in range(len(indices))]

    df_res['phi_liquido_promethee'] = phi_net
    df_res['rank_promethee'] = (-phi_net).argsort().argsort() + 1
    df_res['escolhida_promethee'] = [i == idx_vencedor_promethee for i in range(len(indices))]

    df_res['score_ahp_inverso'] = scores_inv
    df_res['rank_ahp_inverso'] = (-scores_inv).argsort().argsort() + 1
    df_res['escolhida_ahp_inverso'] = [i == idx_vencedor_ahp_inv for i in range(len(indices))]

    arquivo_csv = "tabela_tomada_decisao.csv"
    df_res.to_csv(arquivo_csv, sep=";", index=False, encoding="utf-8-sig")
    print(f"Tabela de decisão salva em: {arquivo_csv}")

    # Exibir no terminal
    print(f"\nRazão de Consistência AHP (CR): {cr:.4f}")
    print(f"Pesos AHP (f1..f4): {np.round(pesos_ahp, 4)}")
    print(f"Vencedoras -> AHP(faixa): Alt {idx_vencedor_ahp+1} | AHP(inverso): Alt {idx_vencedor_ahp_inv+1} "
          f"| PROMETHEE II: Alt {idx_vencedor_promethee+1}")
    print("\nMatriz de Decisão e Rankings das Alternativas Selecionadas:")
    print(df_res.to_string())

    # 6. Gerar as Figuras das Letras (e) e (f)
    plot_projecoes_pareto(matriz_pareto, matriz_decisao, idx_vencedor_ahp, idx_vencedor_promethee)
    
    X_ahp_best = alts_X[idx_vencedor_ahp]
    X_prom_best = alts_X[idx_vencedor_promethee]
    
    plot_caracterizacao_solucao(X_ahp_best, f"Alternativa {idx_vencedor_ahp+1} (Recomendação AHP)", "solucao_final_ahp.png")
    plot_caracterizacao_solucao(X_prom_best, f"Alternativa {idx_vencedor_promethee+1} (Recomendação PROMETHEE)", "solucao_final_promethee.png")