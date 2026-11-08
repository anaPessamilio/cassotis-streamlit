# =============================================================
# model.py — App Streamlit do Case Cassotis (Entrega 3: decisão multicritério)
# Rodar com:  streamlit run model.py
#
# Arquivos esperados na mesma pasta (veja SETUP):
#   dados.py, avaliacao.py, mcda.py, pareto.py        (código)
#   solucoes_soma_ponderada.npz, solucoes_epsilon_restrito.npz   (Entrega 2)
#   referencias.json, resultados_execucoes.csv                   (Entregas 1 e 2)
#   resultados_soma_ponderada.csv, resultados_epsilon_restrito.csv (opcionais)
#   tabela_tomada_decisao.csv (opcional: o app recalcula tudo a partir dos .npz)
# =============================================================
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

import avaliacao as A
import dados as D
from mcda import (
    carregar_fronteira_global_unificada,
    executar_ahp,
    executar_promethee_ii,
)

# =========================================================================================================
# CONFIG
# =========================================================================================================
ARQ_CASE_PDF = "Case_UFMG2026.pdf"
ARQ_DECISAO_CSV = "tabela_tomada_decisao.csv"
ARQ_SP = "solucoes_soma_ponderada.npz"
ARQ_EPS = "solucoes_epsilon_restrito.npz"
ARQ_REF = "referencias.json"
ARQ_EXEC = "resultados_execucoes.csv"
ARQ_RES_SP = "resultados_soma_ponderada.csv"
ARQ_RES_EPS = "resultados_epsilon_restrito.csv"
ARQ_LOGO_SIDEBAR = "side_bar_logo.png"
ARQ_LOGO_MAIN = "main_body_logo.png"

N_ALTERNATIVAS = 8

# ATENÇÃO: estes parâmetros devem ser IGUAIS aos do bloco __main__ do mcda.py
SAATY = np.array([
    [1.0, 3.0, 3.0, 5.0],
    [1/3, 1.0, 1.0, 3.0],
    [1/3, 1.0, 1.0, 3.0],
    [1/5, 1/3, 1/3, 1.0],
])
Q_LIM = np.array([500000.0, 0.20, 0.15, 10.0])    # limiar de indiferença
P_LIM = np.array([2000000.0, 1.00, 0.80, 50.0])   # limiar de preferência
Q_FRAC, P_FRAC = 0.05, 0.25                       # versão proporcional à faixa de referência

# Limites inferiores EXATOS (gerados por limites_exatos.py: MILP para f1, relaxação contínua para f2/f3)
LIMITES_EXATOS = {"f1": 59_360_000.0, "f2": 2.3725, "f3": 1.4199}

COLS_CRIT = ["f1_custo_R$", "f2_desvio_sio2", "f3_desvio_al2o3", "f4_homogeneidade"]
NOMES_CRIT = ["f1 · Custo", "f2 · Desvio SiO₂", "f3 · Desvio Al₂O₃", "f4 · Homogeneidade"]
CURTO_CRIT = ["Custo", "SiO₂", "Al₂O₃", "Homogeneidade"]

MODOS_AHP = {
    "faixa_ref": "Faixa de referência da Entrega 1 — adotada",
    "faixa_front": "Faixa da fronteira global",
    "inverso": "Inverso, 1/valor (teste de sensibilidade)",
}
MODOS_PROM = {
    "prop": f"Proporcionais à faixa de referência (q = {Q_FRAC:.0%}, p = {P_FRAC:.0%}) — adotada",
    "orig": "Absolutos (teste de sensibilidade)",
}

BLUE, ORANGE, AQUA, RED, GOLD, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#d6336c", "#e0a800", "#9aa0a6"
PALETA_ALT = ["#2a78d6", "#eb6834", "#1baf7a", "#d6336c", "#e0a800", "#7a5cc7", "#17a2b8", "#6c757d"]


# =========================================================================================================
# FORMATAÇÃO
# =========================================================================================================
def br(v, d=2):
    """Número no padrão brasileiro (1.234,56)."""
    s = f"{v:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def mi(v, d=2):
    return br(v / 1e6, d)


# =========================================================================================================
# CARGA DE DADOS (cacheada)
# =========================================================================================================
@st.cache_data
def carregar_csv_decisao():
    if not os.path.exists(ARQ_DECISAO_CSV):
        return None
    return pd.read_csv(ARQ_DECISAO_CSV, sep=";", encoding="utf-8-sig")


@st.cache_data
def carregar_execucoes():
    if not os.path.exists(ARQ_EXEC):
        return None
    return pd.read_csv(ARQ_EXEC, sep=";", encoding="utf-8-sig")


@st.cache_data
def carregar_csv_multi(arq):
    if not os.path.exists(arq):
        return None
    return pd.read_csv(arq, sep=";", encoding="utf-8-sig")


@st.cache_data
def carregar_referencias():
    """Valores isolados da Entrega 1 (mín/máx por objetivo). Retorna (lo[3], hi[3], origem) ou None."""
    if os.path.exists(ARQ_REF):
        with open(ARQ_REF, "r", encoding="utf-8") as f:
            r = json.load(f)
        lo = np.array([r["f1"]["min"], r["f2"]["min"], r["f3"]["min"]], dtype=float)
        hi = np.array([r["f1"]["max"], r["f2"]["max"], r["f3"]["max"]], dtype=float)
        return lo, hi, ARQ_REF
    ex = carregar_execucoes()
    if ex is not None:
        cols = ["f1_custo_R$", "f2_desvio_sio2", "f3_desvio_al2o3"]
        lo = np.array([ex.loc[ex.objetivo_otimizado == o, c].min()
                       for o, c in zip(("f1_custo", "f2_sio2", "f3_al2o3"), cols)])
        hi = ex[cols].max().to_numpy(dtype=float)
        return lo, hi, ARQ_EXEC
    return None


@st.cache_data(show_spinner="Carregando a fronteira de Pareto unificada...")
def carregar_fronteira():
    """Retorna (X_pareto, matriz_pareto, indices_das_alternativas) ou None se faltarem os .npz."""
    if not (os.path.exists(ARQ_SP) and os.path.exists(ARQ_EPS)):
        return None
    X_p, M_p = carregar_fronteira_global_unificada(ARQ_SP, ARQ_EPS)
    m_min, m_max = M_p.min(axis=0), M_p.max(axis=0)
    m_norm = (M_p - m_min) / np.where(m_max - m_min == 0, 1, m_max - m_min)
    idx = np.argsort(np.sqrt(np.sum(m_norm ** 2, axis=1)))[:N_ALTERNATIVAS]
    return X_p, M_p, idx


# =========================================================================================================
# MÉTODOS MULTICRITÉRIO (com normalização escolhida)
# =========================================================================================================
def faixa_modo(modo, M, M_glob, ref):
    """Faixa (lo, hi) por critério para a normalização do AHP. Retorna None para o modo 'inverso'."""
    if modo == "inverso":
        return None
    base = M_glob if M_glob is not None else M
    lo, hi = base.min(axis=0).astype(float), base.max(axis=0).astype(float)
    if modo == "faixa_ref" and ref is not None:
        lo[:3], hi[:3] = ref[0], ref[1]       # f4 não tem referência na Entrega 1 -> usa a fronteira
    return lo, hi


def ahp_scores(M, w, faixa):
    """AHP (minimização). faixa=None -> normalização pelo inverso; senão min-max linear pela faixa."""
    if faixa is None:
        inv = 1.0 / M
        norm = inv / inv.sum(axis=0, keepdims=True)
    else:
        lo, hi = faixa
        norm = (hi - M) / (hi - lo)
    return norm @ w


def limiares(modo, faixa_ref_):
    if modo == "prop":
        lo, hi = faixa_ref_
        return P_FRAC * (hi - lo), Q_FRAC * (hi - lo)
    return P_LIM, Q_LIM


def promethee(M, w, p_lim, q_lim):
    return executar_promethee_ii(M, w, p_lim, q_lim)[0]


def spearman(r1, r2):
    return float(np.corrcoef(np.asarray(r1, float), np.asarray(r2, float))[0, 1])


def pesos_com_custo(w0, wc):
    """Pesos com o custo fixado em wc; os demais mantêm as proporções do AHP."""
    return np.concatenate([[wc], (1 - wc) * w0[1:] / w0[1:].sum()])


@st.cache_data
def varredura_custo(M, w0, faixa, p_lim, q_lim, passo=0.01):
    grade = np.round(np.arange(0.0, 1.0 + 1e-9, passo), 4)
    v_ahp = np.zeros(len(grade), dtype=int)
    v_prom = np.zeros(len(grade), dtype=int)
    for k, wc in enumerate(grade):
        w = pesos_com_custo(w0, wc)
        v_ahp[k] = int(np.argmax(ahp_scores(M, w, faixa)))
        v_prom[k] = int(np.argmax(promethee(M, w, p_lim, q_lim)))
    return grade, v_ahp, v_prom


@st.cache_data
def robustez_pesos(M, w0, faixa, p_lim, q_lim, conc=30, n=2000, seed=0):
    """Monte Carlo LOCAL: pesos ~ Dirichlet(w0 * conc), centrados nos pesos de Saaty.
    conc=30 -> desvio-padrão do peso de custo ~9 p.p.; conc=100 -> ~5 p.p. Retorna % de vitórias por alternativa."""
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.asarray(w0) * conc, size=n)
    v_ahp, v_prom = np.zeros(len(M)), np.zeros(len(M))
    for w in W:
        v_ahp[int(np.argmax(ahp_scores(M, w, faixa)))] += 1
        v_prom[int(np.argmax(promethee(M, w, p_lim, q_lim)))] += 1
    return v_ahp / n * 100, v_prom / n * 100


def folga_operacional(X):
    """Menor distância (em p.p.) entre o teor de cada pilha e o limite mais próximo (mín/máx), para SiO₂ e Al₂O₃."""
    out = []
    for ind, lo, hi in ((D.sio2_m, D.min_sio2, D.max_sio2), (D.al2o3_m, D.min_al2o3, D.max_al2o3)):
        q = A.teor_pilha(X, ind)
        out.append(float(np.min(np.minimum(q - lo, hi - q))))
    return out


def intervalos(grade, vencedoras):
    """Trechos contínuos [início, fim] do peso de custo em que cada alternativa vence."""
    out, ini = [], 0
    for k in range(1, len(grade) + 1):
        if k == len(grade) or vencedoras[k] != vencedoras[ini]:
            out.append((float(grade[ini]), float(grade[k - 1]), int(vencedoras[ini])))
            ini = k
    return out


def decomp_vies(X, indicador, alvo):
    """% do desvio quadrático (f2 ou f3) que é VIÉS da média do equipamento (e não variação entre pilhas)."""
    q = A.teor_pilha(X, indicador)
    total = float(np.sum((q - alvo) ** 2))
    vies = sum(len(g) * (q[g].mean() - alvo[g[0]]) ** 2 for g in (D.IDX_S1, D.IDX_S2))
    return total, (vies / total * 100 if total > 0 else 0.0)


# =========================================================================================================
# GRÁFICOS
# =========================================================================================================
def layout_base(fig, h=380, **kw):
    kw.setdefault("margin", dict(t=50, b=40))
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=h, **kw)
    fig.update_yaxes(gridcolor="#e5e5e5", gridwidth=1, zeroline=False)
    return fig


def fig_desvio_alvo(sinter, qi, cor):
    alvo = D.SPECS[sinter][f"{qi}_alvo"]
    mins = [m for m in D.NOMES_MINERIOS if D.MINERIOS[m][f"S{sinter}"]]
    desvios = [D.MINERIOS[m][qi] - alvo for m in mins]
    fig = go.Figure(go.Bar(
        x=mins, y=desvios, marker_color=[AQUA if d < 0 else cor for d in desvios],
        text=[f"{d:+.2f}" for d in desvios], textposition="outside", textfont_size=10,
    ))
    fig.add_hline(y=0, line_width=1.5, line_color="#333")
    layout_base(fig, 350, title=f"{qi} — Alvo Sinter {sinter}: {alvo}%",
                yaxis_title=f"Desvio {qi} (p.p.)", xaxis_title="Minério")
    return fig


def fig_projecao(M_glob, M_alt, i_ahp, i_prom, ix, iy, ref_lo=None):
    esc = [1e6, 1, 1, 1]
    rot = ["Custo (R$ milhões)", "Σ desvio² SiO₂", "Σ desvio² Al₂O₃", "Homogeneidade (f4)"]
    fig = go.Figure()
    if M_glob is not None:
        fig.add_trace(go.Scatter(x=M_glob[:, ix] / esc[ix], y=M_glob[:, iy] / esc[iy], mode="markers",
                                 marker=dict(color=GRAY, size=6, opacity=0.4), name="Fronteira global"))
    fig.add_trace(go.Scatter(
        x=M_alt[:, ix] / esc[ix], y=M_alt[:, iy] / esc[iy], mode="markers+text",
        text=[f"A{i + 1}" for i in range(len(M_alt))], textposition="top center",
        marker=dict(color=BLUE, size=10), name=f"Alternativas ({len(M_alt)})"))
    if ref_lo is not None and ix < 3 and iy < 3:
        fig.add_trace(go.Scatter(x=[ref_lo[ix] / esc[ix]], y=[ref_lo[iy] / esc[iy]], mode="markers",
                                 marker=dict(color="rgba(0,0,0,0)", size=12, symbol="x", line=dict(width=2, color="#333")),
                                 name="Melhor da Entrega 1"))
    if i_ahp == i_prom:
        fig.add_trace(go.Scatter(x=[M_alt[i_ahp, ix] / esc[ix]], y=[M_alt[i_ahp, iy] / esc[iy]], mode="markers",
                                 marker=dict(color=RED, size=18, symbol="star", line=dict(width=1.5, color="black")),
                                 name=f"AHP e PROMETHEE (A{i_ahp + 1})"))
    else:
        fig.add_trace(go.Scatter(x=[M_alt[i_ahp, ix] / esc[ix]], y=[M_alt[i_ahp, iy] / esc[iy]], mode="markers",
                                 marker=dict(color=RED, size=18, symbol="star"), name=f"AHP (A{i_ahp + 1})"))
        fig.add_trace(go.Scatter(x=[M_alt[i_prom, ix] / esc[ix]], y=[M_alt[i_prom, iy] / esc[iy]], mode="markers",
                                 marker=dict(color=GOLD, size=14, symbol="diamond", line=dict(width=1, color="black")),
                                 name=f"PROMETHEE (A{i_prom + 1})"))
    layout_base(fig, 430, xaxis_title=rot[ix], yaxis_title=rot[iy],
                legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
    fig.update_xaxes(gridcolor="#e5e5e5")
    return fig


def fig_composicao(X):
    nomes = [p["nome"] for p in D.PILHAS]
    fig = go.Figure()
    for i, m in enumerate(D.NOMES_MINERIOS):
        if X[i, :].sum() == 0:
            continue
        fig.add_trace(go.Bar(x=nomes, y=X[i, :] * 2.0, name=m))
    layout_base(fig, 420, barmode="stack", yaxis_title="Massa (kt)", title="Composição das pilhas por minério (kt)")
    return fig


def fig_qualidade(X, indicador, nome, alvo, mn, mx, cor):
    """Teor por pilha (pontos) contra faixa mín/alvo/máx de cada equipamento, com eixo ajustado à faixa."""
    q = A.teor_pilha(X, indicador)
    nomes = [p["nome"] for p in D.PILHAS]
    dentro = (q >= mn - 1e-9) & (q <= mx + 1e-9)
    fig = go.Figure()
    estilos = (("Máximo", mx, dict(color="crimson", dash="dot")),
               ("Alvo", alvo, dict(color="green", dash="dash", width=2)),
               ("Mínimo", mn, dict(color="crimson", dash="dot")))
    for k, grupo in enumerate((D.IDX_S1, D.IDX_S2)):
        for nm, arr, ln in estilos:
            fig.add_trace(go.Scatter(x=[nomes[p] for p in grupo], y=[arr[p] for p in grupo], mode="lines",
                                     line=ln, name=nm, legendgroup=nm, showlegend=(k == 0)))
    fig.add_trace(go.Scatter(
        x=nomes, y=q, mode="markers+text", text=[f"{v:.2f}" for v in q], textposition="bottom center",
        textfont_size=10, name=f"{nome} da pilha",
        marker=dict(size=12, color=[cor if d else RED for d in dentro], line=dict(width=1, color="#222"))))
    lo, hi = min(mn.min(), q.min()), max(mx.max(), q.max())
    pad = 0.12 * (hi - lo)
    layout_base(fig, 420, title=f"Teor de {nome} por pilha (%)", yaxis_title=f"{nome} (%)",
                legend=dict(orientation="h", yanchor="top", y=-0.18, x=0), margin=dict(t=50, b=90))
    fig.update_yaxes(range=[lo - pad, hi + pad])
    return fig


def fig_heatmap_receitas(X, sinter):
    idx = [p for p, info in enumerate(D.PILHAS) if info["sinter"] == sinter]
    sub = X[:, idx].astype(float)
    prop = sub / sub.sum(axis=0, keepdims=True) * 100
    linhas = np.flatnonzero(sub.sum(axis=1) > 0)
    fig = go.Figure(go.Heatmap(
        z=prop[linhas], x=[D.PILHAS[p]["nome"] for p in idx], y=[D.NOMES_MINERIOS[i] for i in linhas],
        colorscale="Blues", texttemplate="%{z:.0f}%", colorbar=dict(title="% da pilha")))
    fig.update_yaxes(type="category", autorange="reversed")
    fig.update_layout(title=f"Receita (% de caminhões por minério) — Sinter {sinter}",
                      height=max(260, 28 * len(linhas) + 120), margin=dict(t=40, b=30))
    return fig


def fig_peso_efetivo(M, w, faixas):
    """Peso declarado x influência efetiva de cada critério (peso × amplitude normalizada entre as alternativas)."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=CURTO_CRIT, y=w * 100, name="Peso declarado (Saaty)", marker_color=GRAY))
    cores = {"inverso": ORANGE, "faixa_ref": BLUE, "faixa_front": AQUA}
    for modo, fx in faixas.items():
        if fx is None:
            inv = 1.0 / M
            norm = inv / inv.sum(axis=0, keepdims=True)
        else:
            norm = (fx[1] - M) / (fx[1] - fx[0])
        amp = w * (norm.max(axis=0) - norm.min(axis=0))
        ef = amp / amp.sum() * 100
        fig.add_trace(go.Bar(x=CURTO_CRIT, y=ef, name=f"Influência efetiva — {MODOS_AHP[modo].split(' (')[0]}",
                             marker_color=cores[modo], text=[f"{v:.0f}%" for v in ef], textposition="outside"))
    layout_base(fig, 380, barmode="group", yaxis_title="% do resultado do AHP",
                title="O peso declarado de cada critério chega ao ranking?",
                legend=dict(orientation="h", yanchor="top", y=-0.15, x=0), margin=dict(t=50, b=90))
    return fig


def fig_intervalos(grade, v_ahp, v_prom, w0_custo):
    fig = go.Figure()
    mostrados = set()
    for nome, v in (("PROMETHEE II", v_prom), ("AHP", v_ahp)):
        for ini, fim, alt in intervalos(grade, v):
            fig.add_trace(go.Bar(
                y=[nome], x=[(fim - ini) * 100 + 1], base=[ini * 100], orientation="h",
                marker_color=PALETA_ALT[alt % len(PALETA_ALT)], name=f"Alt {alt + 1}",
                legendgroup=f"a{alt}", showlegend=alt not in mostrados,
                text=[f"A{alt + 1}"], textposition="inside", hovertemplate=f"Alt {alt + 1}: {ini:.0%}–{fim:.0%}<extra></extra>"))
            mostrados.add(alt)
    fig.add_vline(x=w0_custo * 100, line_dash="dash", line_color="#333",
                  annotation_text=f"peso declarado: {w0_custo:.1%}", annotation_position="top")
    layout_base(fig, 260, barmode="overlay", xaxis_title="Peso do critério Custo (%) — os demais mantêm as proporções do AHP",
                margin=dict(t=60, b=50), legend=dict(orientation="h", yanchor="top", y=-0.35, x=0))
    fig.update_xaxes(range=[0, 101], gridcolor="#e5e5e5")
    return fig


# =========================================================================================================
# FRONT-END
# =========================================================================================================
st.set_page_config(layout="wide", page_title="Case Otimização UFMG")
if os.path.exists(ARQ_LOGO_SIDEBAR) and os.path.exists(ARQ_LOGO_MAIN):
    st.logo(ARQ_LOGO_SIDEBAR, link="https://www.cassotis.com/", icon_image=ARQ_LOGO_MAIN, size="large")
st.title("Case: Otimização de custo e qualidade na formação de pilhas de minério")
st.subheader("Cassotis Consulting - Engenharia de Sistemas UFMG 2026.2")
st.divider()

with st.sidebar:
    st.title("Menu Principal")
    aba_selecionada = st.radio(
        "Ir para:", ["Material auxiliar", "Dados de Entrada", "Decisões de Modelagem", "Resultados"]
    )
    st.divider()
    st.caption("Material desenvolvido e elaborado por Cassotis Consulting")


# ---------------------------------------------------------------------------------------------------------
if aba_selecionada == "Material auxiliar":
    st.header("Material Auxiliar")
    st.markdown("Apresentação do case:")
    if os.path.exists(ARQ_CASE_PDF):
        pdf_viewer(ARQ_CASE_PDF)
    else:
        st.error(f"Arquivo '{ARQ_CASE_PDF}' não encontrado na pasta do app.")

# ---------------------------------------------------------------------------------------------------------
elif aba_selecionada == "Dados de Entrada":
    st.header("Minérios")
    rows = []
    for m, v in D.MINERIOS.items():
        rows.append({
            "Minério": m, "Preço (R$/t)": v["preco"], "FeT (%)": v["FeT"],
            "SiO2 (%)": v["SiO2"], "Al2O3 (%)": v["Al2O3"],
            "Disp. mín (n° de caminhões)": v["dmin"], "Disp. máx (n° de caminhões)": v["dmax"],
            "Sinter 1": "☒" if v["S1"] else "☐", "Sinter 2": "☒" if v["S2"] else "☐",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Minério"), width="stretch")

    for s in (1, 2):
        st.header(f"Sinter {s}")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Especificações de Qualidade")
            sp = D.SPECS[s]
            st.dataframe(pd.DataFrame(
                {"SiO2 (%)": [sp["SiO2_min"], sp["SiO2_alvo"], sp["SiO2_max"]],
                 "Al2O3 (%)": [sp["Al2O3_min"], sp["Al2O3_alvo"], sp["Al2O3_max"]]},
                index=["Mínimo", "Alvo", "Máximo"]), width="stretch")
        with c2:
            st.subheader("Pilhas")
            st.dataframe(pd.DataFrame(
                [{"Equipamento": f"Sinter {s}", "Pilha": p["nome"], "Massa da pilha (kt)": p["massa_kt"],
                  "Caminhões": p["n_cam"]} for p in D.PILHAS if p["sinter"] == s]),
                hide_index=True, width="stretch")

    st.header("Visualizações")
    st.subheader("Preço por Minério")
    precos = [D.MINERIOS[m]["preco"] for m in D.NOMES_MINERIOS]
    fig = go.Figure(go.Bar(x=D.NOMES_MINERIOS, y=precos, marker_color=BLUE, text=precos,
                           textposition="outside", textfont_size=11))
    st.plotly_chart(layout_base(fig, 380, yaxis_title="Preço (R$/t)", xaxis_title="Minério"), width="stretch")

    st.subheader("Disponibilidade por Minério (nº de caminhões)")
    dmn = [D.MINERIOS[m]["dmin"] for m in D.NOMES_MINERIOS]
    dmx = [D.MINERIOS[m]["dmax"] for m in D.NOMES_MINERIOS]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=D.NOMES_MINERIOS, y=dmn, name="Mínimo", marker_color=BLUE))
    fig.add_trace(go.Bar(x=D.NOMES_MINERIOS, y=[b - a for a, b in zip(dmn, dmx)], name="Faixa disponível",
                         marker_color=AQUA, text=dmx, textposition="outside", textfont_size=10))
    st.plotly_chart(layout_base(fig, 380, barmode="stack", yaxis_title="Nº de caminhões", xaxis_title="Minério",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)),
                    width="stretch")

    for s in (1, 2):
        st.subheader(f"Desvio da Qualidade vs. Alvo — Sinter {s}")
        for qi, cor in (("SiO2", BLUE), ("Al2O3", ORANGE)):
            st.plotly_chart(fig_desvio_alvo(s, qi, cor), width="stretch")

# ---------------------------------------------------------------------------------------------------------
elif aba_selecionada == "Decisões de Modelagem":
    st.header("Decisões de Modelagem")

    _, pesos_txt, cr_txt = executar_ahp(np.ones((2, 4)), SAATY)
    ex_txt = carregar_execucoes()
    if ex_txt is not None:
        p_it, p_bl, p_ts = (int(ex_txt.iteracoes_vns.iloc[0]), int(ex_txt.iteracoes_busca_local_max.iloc[0]),
                            int(ex_txt.tentativas_sem_melhoria_max.iloc[0]))
        seeds_txt = ", ".join(str(s) for s in sorted(ex_txt.semente.unique()))
    else:
        p_it, p_bl, p_ts, seeds_txt = 1500, 20, 10, "42, 7, 13, 99, 2024"

    st.subheader("1. Codificação do Problema")
    st.markdown(r"""
Cada solução é uma **matriz inteira** $X \in \mathbb{Z}_{\ge 0}^{20 \times 10}$, onde $x_{i,p}$ é o número de
**caminhões** do minério $i$ enviados à pilha $p$ (cada caminhão = 2 kt). Em `numpy`: `X.shape == (20, 10)`, `dtype=int`.

- Medir em caminhões embute a restrição de **múltiplos de 2 kt** na própria variável.
- Pilhas 1–5 alimentam o Sinter 1 e pilhas 6–10 o Sinter 2; a **elegibilidade** é uma máscara booleana `elegivel[i, p]`.
- O teor de uma pilha é a média ponderada $q_p = \sum_i x_{i,p}\, a_i \,/\, \sum_i x_{i,p}$, calculada de forma
  vetorial (`indicador @ X / X.sum(axis=0)`), o que deixa a avaliação barata dentro do VNS.
- Frente à codificação em vetor de posições sugerida no enunciado, a matriz de contagens não distingue a *ordem* dos
  caminhões — que é irrelevante, pois a pilha homogeneíza em camadas.
""")

    st.subheader("2. Estruturas de Vizinhança")
    st.markdown(r"""
| Vizinhança | Movimento | Por que existe |
|---|---|---|
| **N1 — troca simples** | Em uma pilha sorteada, troca 1 caminhão de um minério presente por outro **elegível** | Ajuste fino de custo e qualidade de uma pilha só |
| **N2 — troca entre pilhas** | Troca 1 caminhão $i$ de $p_1$ por 1 caminhão $j$ de $p_2$ (só se ambos forem elegíveis nos dois sentidos) | **Preserva o consumo total** de cada minério, logo não mexe na disponibilidade |
| **N3 — guiada por custo** | Na pilha sorteada, troca **todos** os caminhões do minério mais caro presente pelo elegível mais barato | Lógica distinta de N1 (movimento em bloco, direcionado): dá saltos de custo que N1 levaria muitas iterações para fazer |

**Tratamento de inviáveis — política de penalização.** N1, N2 e N3 preservam por construção a *massa de cada pilha* e a
*elegibilidade*. Podem violar *qualidade* e *disponibilidade* (e a solução inicial pode violar a disponibilidade quando o
`dmax` não comporta a pilha). Em vez de rejeitar o movimento — o que prenderia a busca, porque é preciso atravessar
regiões inviáveis —, a função avaliada é

$$ f_k^{pen}(X) = f_k(X) + s_k\,\big(100\,V_{massa} + 50\,V_{qualidade} + 100\,V_{elegib.} + 50\,V_{disp.}\big) $$

com $s_k = 10^6$ para $f_1$ (que está na casa de $10^8$) e $s_k = 1$ para $f_2, f_3$. O VNS guarda separadamente a
**melhor solução viável** encontrada, de modo que a penalidade orienta a busca mas nunca contamina o resultado final.
Os pesos 100/50 foram escolhidos por tentativa, e **a sensibilidade a eles não foi quantificada**.
""")

    st.subheader("3. Heurística Construtiva (Solução Inicial)")
    st.markdown("""
Construção gulosa por preço, como sugerido no enunciado:

1. **Mínimos obrigatórios:** para cada minério com `dmin > 0` (M4, M10, M11, M20), distribui os caminhões mínimos entre
   as pilhas elegíveis, em ordem aleatória (a semente dá diversidade entre execuções).
2. **Completar pelas pilhas:** para cada pilha, preenche o que falta com os minérios elegíveis **do mais barato para o
   mais caro**, respeitando o `dmax` global de cada minério.
3. Se ainda faltar massa por limite de disponibilidade, despeja o restante no minério elegível mais barato; a
   inviabilidade resultante é medida e penalizada — não escondida.

A heurística ignora a qualidade de propósito: parte do menor custo possível e deixa o VNS "comprar" qualidade ao longo da busca.
""")
    st.markdown(f"**Parâmetros do VNS (declarados):** {p_it} iterações por execução · busca local com até {p_bl} movimentos "
                f"amostrados, encerrando após {p_ts} sem melhora · k máx. = 3 · sementes {seeds_txt}.")

    st.subheader("4. Métodos de Auxílio à Tomada de Decisão")
    st.markdown(f"""
**Alternativas.** A fronteira global é a união das soluções não dominadas da Soma Ponderada e do ε-Restrito,
re-filtradas por não dominância cruzada (em f1, f2, f3). Dela saem **{N_ALTERNATIVAS} alternativas**: as de menor distância
euclidiana ao ponto utópico (0,0,0,0) após normalização min-max — uma amostra de compromisso, que evita decidir sobre
centenas de pontos quase idênticos.

**Dois métodos de famílias diferentes**, para verificar que a escolha não é um artefato de uma única regra de agregação (a concordância testa a agregação; **não** valida os pesos):

- **AHP** (agregação aditiva, totalmente compensatória). Os pesos vêm de uma matriz de Saaty 4×4 (autovetor principal;
  razão de consistência **CR = {cr_txt:.3f}**, abaixo do limite de 0,10).
- **PROMETHEE II, função tipo V** (superação, compensação parcial). Usa os **mesmos pesos** do AHP, mas compara as
  alternativas par a par com limiar de indiferença *q* e de preferência *p*.

ELECTRE foi descartado por produzir, em geral, uma relação parcial (núcleo) menos direta de comparar com o ranking completo do AHP.

**Normalização do AHP.** Cada critério é normalizado por **min-max linear pela faixa [melhor, pior] dos valores isolados da
Entrega 1** para f1–f3 (f4 não tem referência na Entrega 1 e usa a faixa da fronteira global). Assim, a escala reflete a
amplitude real de cada critério e os pesos declarados determinam a influência de cada um no ranking. A aba *Resultados*
permite comparar com a faixa da fronteira global e com a normalização pelo inverso (1/valor), como teste de sensibilidade.

**Limiares do PROMETHEE.** São **proporcionais à faixa de referência de cada critério: q = {Q_FRAC:.0%} e p = {P_FRAC:.0%}**,
o que trata todos os critérios da mesma forma e preserva os pesos declarados. Limiares absolutos (tabela abaixo) são
usados apenas como teste de sensibilidade.

**Justificativa da matriz de Saaty.** Os julgamentos (escala 1–9) seguem a lógica do problema:

- **Custo ≻ SiO₂ e Al₂O₃ (3, "moderadamente mais importante").** O custo é o ganho econômico direto e mensurável da decisão de compra.
  Os teores já são protegidos por restrições rígidas (faixas mín/máx por pilha), então o desvio em relação ao alvo é um
  refinamento dentro da região viável, não uma condição de viabilidade. Não é 5 porque a qualidade do sinter é o motivo de existir
  do problema: o desvio custa dinheiro mais adiante, na dosagem de fundente e na escória.
- **SiO₂ = Al₂O₃ (1).** Os dois entram na especificação de cada sinter com o mesmo tipo de métrica (desvio quadrático em relação ao alvo)
  e o enunciado não hierarquiza um sobre o outro; qualquer diferença seria arbitrária.
- **Custo ≻ homogeneidade (5, "fortemente mais importante").** f4 é um ganho operacional de manuseio no pátio, indireto, sem
  contrapartida financeira direta nem restrição associada.
- **SiO₂ e Al₂O₃ ≻ homogeneidade (3).** A qualidade tem especificação a cumprir; a homogeneidade de receitas é conveniência, e é
  fracamente correlacionada com os demais critérios (aba 2).
- **Consistência.** A cadeia custo → qualidade → homogeneidade (3 e 3) implicaria 9 entre custo e f4; o grupo declarou 5, uma pequena incoerência aceitável, refletida em **CR = {cr_txt:.3f}** (limite: 0,10).

Os pesos resultantes (≈ 52% / 20% / 20% / 8%) são um **julgamento do grupo, sem validação externa**. Por isso a aba
*Resultados* testa explicitamente o que acontece se o julgamento principal (custo vs. qualidade) for 1, 2, 4 ou 5.
""")
    st.markdown("**Matriz de Saaty (linhas/colunas = f1, f2, f3, f4) e pesos resultantes**")
    cs1, cs2 = st.columns([3, 2])
    with cs1:
        st.dataframe(pd.DataFrame(SAATY, index=CURTO_CRIT, columns=CURTO_CRIT).round(3), width="stretch")
    with cs2:
        st.dataframe(pd.DataFrame({"Peso": pesos_txt}, index=CURTO_CRIT).style.format("{:.1%}"), width="stretch")
    st.markdown("**Limiares absolutos do PROMETHEE II (usados apenas como teste de sensibilidade)**")
    st.dataframe(pd.DataFrame({"q (indiferença)": Q_LIM, "p (preferência)": P_LIM}, index=NOMES_CRIT), width="stretch")

    st.subheader("5. Atributos de Interesse")
    st.markdown(r"""
| Critério | Definição | Sentido | Por que entra |
|---|---|---|---|
| **f1 — Custo** | $2000\sum_{i,p} x_{i,p}\,c_i$ (R$) | minimizar | É o ganho econômico direto da decisão de compra |
| **f2 — Desvio de SiO₂** | $\sum_p (q^{SiO_2}_p - alvo_p)^2$ | minimizar | Variação de sílica desregula a dosagem de fundente |
| **f3 — Desvio de Al₂O₃** | $\sum_p (q^{Al_2O_3}_p - alvo_p)^2$ | minimizar | Alumina eleva a viscosidade da escória; é o indicador que mais penaliza minérios baratos |
| **f4 — Homogeneidade de receita** | $\sum_{g}\sum_{p\in g}\sum_i (x_{i,p}-\bar{x}_{i,g})^2$ | minimizar | Mede o afastamento das receitas entre pilhas do mesmo equipamento, em caminhões de cada minério; simplifica o manuseio no pátio |

f4 depende da *composição em caminhões*, não do teor resultante, então **não é uma reescrita de f2 ou f3**. Medido na
fronteira, porém, ele é só **fracamente** conflitante com os demais (veja a correlação na aba *Resultados*), e não é
otimizado pelas metaheurísticas: é calculado depois, sobre as soluções já geradas. Como é medido em caminhões
absolutos e as pilhas têm tamanhos diferentes (10 a 19 caminhões), o valor mínimo possível não é zero.
""")

# ---------------------------------------------------------------------------------------------------------
elif aba_selecionada == "Resultados":
    st.header("Resultados do Modelo — Entrega 3")

    # ------------------------------------------------------------------ carga e conferência
    ref_tuple = carregar_referencias()
    ref = (ref_tuple[0], ref_tuple[1]) if ref_tuple else None
    fr = carregar_fronteira()
    df_csv = carregar_csv_decisao()
    X_alts, M_glob = None, None

    if fr is not None:
        X_p, M_glob, idx_alts = fr
        X_alts, M = X_p[idx_alts], M_glob[idx_alts]
        if df_csv is not None and (len(df_csv) != len(M) or not np.allclose(df_csv[COLS_CRIT].to_numpy(float), M)):
            st.warning(f"`{ARQ_DECISAO_CSV}` não reproduz as soluções dos .npz (execuções diferentes?). "
                       "O app usa os .npz. Rode `python mcda.py` para regenerar o CSV.")
    elif df_csv is not None:
        M = df_csv[COLS_CRIT].to_numpy(dtype=float)
        st.info(f"Sem `{ARQ_SP}`/`{ARQ_EPS}`: usando só as {len(M)} alternativas do CSV. "
                "Fronteira completa, composição das pilhas e normalização pela fronteira ficam limitadas.")
    else:
        st.error("Não encontrei os arquivos `.npz` da Entrega 2 nem o `tabela_tomada_decisao.csv`.")
        st.stop()

    if ref is None:
        st.warning(f"`{ARQ_REF}` não encontrado: a normalização pela referência da Entrega 1 usará a fronteira global.")

    n_alt = len(M)
    rotulos = [f"Alt {i + 1}" for i in range(n_alt)]
    _, w0, cr = executar_ahp(M, SAATY)

    # ------------------------------------------------------------------ configuração da decisão
    with st.expander("Configuração da análise multicritério", expanded=True):
        cA, cB = st.columns(2)
        with cA:
            modo_ahp = st.radio("Normalização do AHP", list(MODOS_AHP), format_func=MODOS_AHP.get, index=0,
                                help="Define como f1…f4 viram notas comparáveis antes de aplicar os pesos.")
        with cB:
            modo_prom = st.radio("Limiares do PROMETHEE II", list(MODOS_PROM), format_func=MODOS_PROM.get, index=0,
                                 help="Padrão: limiares proporcionais à faixa de cada critério, coerentes com os pesos declarados.")
        st.caption("Padrão adotado: normalização pela faixa da Entrega 1 e limiares do PROMETHEE "
                   "proporcionais à faixa. As demais opções existem para teste de sensibilidade; a tabela da aba 4 mostra "
                   "a vencedora em todas as combinações.")

    faixa_sel = faixa_modo(modo_ahp, M, M_glob, ref)
    faixa_ref_ = faixa_modo("faixa_ref", M, M_glob, ref)
    p_sel, q_sel = limiares(modo_prom, faixa_ref_)

    scores = ahp_scores(M, w0, faixa_sel)
    phi = promethee(M, w0, p_sel, q_sel)
    rank_ahp = (-scores).argsort().argsort() + 1
    rank_prom = (-phi).argsort().argsort() + 1
    i_ahp, i_prom = int(np.argmax(scores)), int(np.argmax(phi))
    i_barata = int(np.argmin(M[:, 0]))

    # configuração adotada na decisão final (independe dos botões acima): AHP por faixa de referência + PROMETHEE proporcional
    p_rec, q_rec = limiares("prop", faixa_ref_)
    sc_rec = ahp_scores(M, w0, faixa_ref_)
    phi_rec = promethee(M, w0, p_rec, q_rec)
    i_rec_a, i_rec_p = int(np.argmax(sc_rec)), int(np.argmax(phi_rec))
    i_rec = i_rec_a
    grade_r, va_r, vp_r = varredura_custo(M, w0, faixa_ref_, p_rec, q_rec)
    ia_r, ip_r = intervalos(grade_r, va_r), intervalos(grade_r, vp_r)

    (t_obj, t_par, t_mat, t_rank, t_sens, t_sol, t_crit) = st.tabs([
        "1. Soluções por objetivo", "2. Fronteira de Pareto", "3. Matriz de decisão",
        "4. Ranking AHP × PROMETHEE", "5. Sensibilidade", "6. Detalhe da solução", "7. Decisão final e análise crítica"])

    # ------------------------------------------------------------------ 1. Soluções por objetivo
    with t_obj:
        st.markdown("Resultados da **Entrega 1**: cada objetivo otimizado separadamente (VNS, 5 execuções com sementes "
                    "diferentes). Esses valores são a *referência de normalização* da Entrega 2.")
        ex = carregar_execucoes()
        if ex is None:
            st.info(f"Coloque `{ARQ_EXEC}` na pasta do app para ver as 15 execuções.")
        else:
            objs = [("f1_custo", "f1_custo_R$", "f1", "f1 · Custo (R$)"),
                    ("f2_sio2", "f2_desvio_sio2", "f2", "f2 · Desvio SiO₂"),
                    ("f3_al2o3", "f3_desvio_al2o3", "f3", "f3 · Desvio Al₂O₃")]
            fmt = {"f1": lambda v: f"R$ {mi(v, 2)} mi", "f2": lambda v: br(v, 3), "f3": lambda v: br(v, 3)}
            # melhor valor de cada objetivo na fronteira da Entrega 2 (se disponível)
            melhor_e2 = {"f1": M_glob[:, 0].min(), "f2": M_glob[:, 1].min(), "f3": M_glob[:, 2].min()} if M_glob is not None else {}

            linhas = []
            for obj, col, k, nome in objs:
                s = ex.loc[ex.objetivo_otimizado == obj, col]
                lim = LIMITES_EXATOS[k]
                linha = {"Objetivo": nome, "Melhor (E1)": fmt[k](s.min()), "Média": fmt[k](s.mean()),
                         "Desvio-padrão": fmt[k](s.std()) if k != "f1" else f"R$ {mi(s.std(), 2)} mi",
                         "Pior": fmt[k](s.max()), "Limite inferior": fmt[k](lim),
                         "Gap do melhor (E1)": f"+{(s.min() / lim - 1) * 100:.1f}%"}
                if k in melhor_e2:
                    linha["Melhor na fronteira (E2)"] = fmt[k](melhor_e2[k])
                    linha["Gap (E2)"] = f"+{(melhor_e2[k] / lim - 1) * 100:.1f}%"
                linhas.append(linha)
            st.dataframe(pd.DataFrame(linhas).set_index("Objetivo"), width="stretch")
            st.caption("Limite inferior de f1: **ótimo exato** (programação inteira, `limites_exatos.py`). "
                       "Limites de f2 e f3: relaxação contínua — o ótimo inteiro é igual ou maior, "
                       "então o gap exibido é um **teto**.")

            # gráfico de gap
            fig = go.Figure()
            for obj, col, k, nome in objs:
                s = ex.loc[ex.objetivo_otimizado == obj, col].to_numpy()
                g = (s / LIMITES_EXATOS[k] - 1) * 100
                fig.add_trace(go.Scatter(x=[nome] * len(g), y=g, mode="markers", name="Execuções da Entrega 1",
                                         marker=dict(size=10, color=BLUE, opacity=0.75), legendgroup="e1",
                                         showlegend=(k == "f1")))
                if k in melhor_e2:
                    fig.add_trace(go.Scatter(x=[nome], y=[(melhor_e2[k] / LIMITES_EXATOS[k] - 1) * 100], mode="markers",
                                             name="Melhor da fronteira (Entrega 2)", legendgroup="e2",
                                             marker=dict(size=14, color=ORANGE, symbol="diamond", line=dict(width=1, color="#222")),
                                             showlegend=(k == "f1")))
            layout_base(fig, 380, yaxis_title="Gap acima do limite inferior (%)",
                        title="Quão perto do limite cada heurística chegou",
                        legend=dict(orientation="h", yanchor="top", y=-0.15, x=0), margin=dict(t=50, b=80))
            st.plotly_chart(fig, width="stretch")

            # melhor solução de cada objetivo: valores dos 3 objetivos
            st.markdown("**Melhor execução de cada objetivo — os três objetivos naquela solução**")
            melhores = []
            for obj, col, k, nome in objs:
                sub = ex[ex.objetivo_otimizado == obj]
                r = sub.loc[sub[col].idxmin()]
                melhores.append({"Otimizado": nome, "Semente": int(r.semente), "f1 · Custo (R$ mi)": mi(r["f1_custo_R$"]),
                                 "f2 · SiO₂": br(r["f2_desvio_sio2"], 3), "f3 · Al₂O₃": br(r["f3_desvio_al2o3"], 3),
                                 "Viável": "Sim" if r["viavel"] else "Não"})
            st.dataframe(pd.DataFrame(melhores).set_index("Otimizado"), width="stretch")
            st.caption("Cada solução é boa no seu objetivo e ruim nos outros — é o conflito que justifica o multiobjetivo.")

            with st.expander("Parâmetros experimentais declarados"):
                p0 = ex.iloc[0]
                st.markdown(
                    f"- **Entrega 1:** {ex.execucao.nunique()} execuções por objetivo × 3 objetivos = {len(ex)} execuções; "
                    f"sementes {', '.join(str(s) for s in sorted(ex.semente.unique()))}; {int(p0.iteracoes_vns)} iterações VNS; "
                    f"busca local ≤ {int(p0.iteracoes_busca_local_max)} movimentos, parada após {int(p0.tentativas_sem_melhoria_max)} sem melhora.")
                sp_, ep_ = carregar_csv_multi(ARQ_RES_SP), carregar_csv_multi(ARQ_RES_EPS)
                if sp_ is not None:
                    st.markdown(f"- **Soma ponderada:** {sp_.cfg_id.nunique()} combinações de pesos (passo 0,1) × "
                                f"{sp_.execucao.nunique()} execuções = {len(sp_)}; {int(sp_.nao_dominada.sum())} não dominadas.")
                if ep_ is not None:
                    st.markdown(f"- **ε-restrito:** grade {int(round(ep_.cfg_id.nunique() ** 0.5))}×"
                                f"{int(round(ep_.cfg_id.nunique() ** 0.5))} = {ep_.cfg_id.nunique()} combinações de ε × "
                                f"{ep_.execucao.nunique()} execuções = {len(ep_)}; apenas {ep_.viavel.mean():.0%} viáveis "
                                f"em relação ao ε; {int(ep_.nao_dominada.sum())} não dominadas.")
                if M_glob is not None:
                    st.markdown(f"- **Fronteira global (união, não dominância cruzada em f1, f2, f3):** {len(M_glob)} soluções.")

        if M_glob is not None and ref is not None:
            melh = [("f1", M_glob[:, 0].min(), ref[0][0], "R$ {:,.2f} mi", 1e6), ("f2", M_glob[:, 1].min(), ref[0][1], "{:.3f}", 1),
                    ("f3", M_glob[:, 2].min(), ref[0][2], "{:.3f}", 1)]
            ganhos = [f"{n}: {fm.format(r_ / sc).replace(',', 'X').replace('.', ',').replace('X', '.')} → "
                      f"{fm.format(v / sc).replace(',', 'X').replace('.', ',').replace('X', '.')}"
                      for n, v, r_, fm, sc in melh if v < r_ - 1e-9]
            if ganhos:
                st.info("**A Entrega 2 refinou os melhores valores da Entrega 1.** Ao combinar os objetivos, a fronteira "
                        "alcançou pontos melhores que os extremos isolados (" + "; ".join(ganhos) + "). Por isso a "
                        "normalização usa a faixa da Entrega 1 como escala de referência, e não como ideal teórico.")

    # ------------------------------------------------------------------ 2. Pareto
    with t_par:
        if M_glob is not None:
            st.markdown(f"Fronteira global unificada (Soma Ponderada ∪ ε-Restrito, não dominância cruzada): "
                        f"**{len(M_glob)} soluções**. Em azul, as {n_alt} alternativas levadas à decisão; "
                        "o × marca o melhor valor isolado de cada objetivo na Entrega 1.")
        c1, c2, c3 = st.columns(3)
        ref_lo = ref[0] if ref else None
        for col, (ix, iy) in zip((c1, c2, c3), ((0, 1), (0, 2), (1, 2))):
            with col:
                st.plotly_chart(fig_projecao(M_glob, M, i_ahp, i_prom, ix, iy, ref_lo), width="stretch")
        if M_glob is not None:
            g = go.Figure()
            g.add_trace(go.Scatter3d(x=M_glob[:, 0] / 1e6, y=M_glob[:, 1], z=M_glob[:, 2], mode="markers",
                                     marker=dict(size=3, color=GRAY, opacity=0.4), name="Fronteira global"))
            g.add_trace(go.Scatter3d(x=M[:, 0] / 1e6, y=M[:, 1], z=M[:, 2], mode="markers+text",
                                     text=[f"A{i + 1}" for i in range(n_alt)], marker=dict(size=6, color=BLUE),
                                     name="Alternativas"))
            g.update_layout(height=520, margin=dict(t=20, b=0),
                            scene=dict(xaxis_title="f1 Custo (R$ mi)", yaxis_title="f2 SiO₂", zaxis_title="f3 Al₂O₃"))
            st.plotly_chart(g, width="stretch")

            st.markdown("**Correlação de Spearman entre os critérios, na fronteira global** "
                        "(negativa = conflito; próxima de zero = quase independentes)")
            R = pd.DataFrame(pd.DataFrame(M_glob, columns=CURTO_CRIT).corr(method="spearman").to_numpy(),
                             index=CURTO_CRIT, columns=CURTO_CRIT)
            st.dataframe(R.style.format("{:.2f}").background_gradient(cmap="RdBu", vmin=-1, vmax=1), width="stretch")
            st.caption("f1 conflita fortemente com f3 (e moderadamente com f2). f4 tem correlação fraca com todos: "
                       "não é redundante, mas também não é um conflito forte.")

    # ------------------------------------------------------------------ 3. Matriz de decisão
    with t_mat:
        st.markdown("Todos os critérios são de **minimização**. Cada linha é uma solução não dominada da fronteira.")
        mat = pd.DataFrame(M, columns=NOMES_CRIT, index=rotulos)
        st.dataframe(mat.style.format({NOMES_CRIT[0]: "R$ {:,.0f}", NOMES_CRIT[1]: "{:.3f}",
                                       NOMES_CRIT[2]: "{:.3f}", NOMES_CRIT[3]: "{:.1f}"})
                     .highlight_min(axis=0, color="#cdeedd"), width="stretch")
        st.caption("Em verde, o melhor valor de cada critério. Nenhuma alternativa é a melhor em todos — é o trade-off.")
        amp = (M.max(axis=0) / M.min(axis=0) - 1) * 100
        st.dataframe(pd.DataFrame({"Variação entre as alternativas (máx/mín − 1)": [f"{v:.1f}%" for v in amp]},
                                  index=NOMES_CRIT).T, width="stretch")
        st.caption("O custo varia pouco entre as alternativas; por isso a normalização escolhida importa tanto (aba 4).")

    # ------------------------------------------------------------------ 4. Ranking
    with t_rank:
        m1, m2, m3 = st.columns(3)
        m1.metric("Vencedora AHP", rotulos[i_ahp])
        m2.metric("Vencedora PROMETHEE II", rotulos[i_prom])
        m3.metric("Spearman entre os rankings", f"{spearman(rank_ahp, rank_prom):.2f}")

        st.markdown(f"**Pesos dos critérios (AHP, CR = {cr:.3f})**")
        st.dataframe(pd.DataFrame({"Peso": w0}, index=NOMES_CRIT).T.style.format("{:.1%}"), width="stretch")

        cA_, cP_ = st.columns(2)
        with cA_:
            f = go.Figure(go.Bar(x=rotulos, y=scores, marker_color=[RED if i == i_ahp else BLUE for i in range(n_alt)],
                                 text=[f"{v:.3f}" for v in scores], textposition="outside"))
            st.plotly_chart(layout_base(f, 360, title="AHP — score global", yaxis_title="Score"), width="stretch")
        with cP_:
            f = go.Figure(go.Bar(x=rotulos, y=phi, marker_color=[GOLD if i == i_prom else BLUE for i in range(n_alt)],
                                 text=[f"{v:+.3f}" for v in phi], textposition="outside"))
            f.add_hline(y=0, line_width=1, line_color="#333")
            st.plotly_chart(layout_base(f, 360, title="PROMETHEE II — fluxo líquido Φ", yaxis_title="Φ"),
                            width="stretch")

        st.dataframe(pd.DataFrame({"Score AHP": scores, "Rank AHP": rank_ahp, "Φ líquido": phi, "Rank PROMETHEE": rank_prom},
                                  index=rotulos).sort_values("Rank AHP")
                     .style.format({"Score AHP": "{:.4f}", "Φ líquido": "{:+.4f}"}), width="stretch")

        st.markdown("#### A vencedora depende da normalização e dos limiares?")
        linhas = []
        for modo in MODOS_AHP:
            sc = ahp_scores(M, w0, faixa_modo(modo, M, M_glob, ref))
            o = np.argsort(-sc)
            linhas.append({"Método": "AHP", "Configuração": MODOS_AHP[modo], "Vencedora": rotulos[o[0]],
                           "2ª colocada": rotulos[o[1]]})
        for modo in MODOS_PROM:
            p_, q_ = limiares(modo, faixa_ref_)
            ph = promethee(M, w0, p_, q_)
            o = np.argsort(-ph)
            linhas.append({"Método": "PROMETHEE II", "Configuração": MODOS_PROM[modo], "Vencedora": rotulos[o[0]],
                           "2ª colocada": rotulos[o[1]]})
        tab_cfg = pd.DataFrame(linhas)
        st.dataframe(tab_cfg, hide_index=True, width="stretch")

        st.plotly_chart(fig_peso_efetivo(M, w0, {m: faixa_modo(m, M, M_glob, ref) for m in MODOS_AHP}),
                        width="stretch")
        st.caption("Com a normalização pelo inverso, o custo (52% declarado) tem influência efetiva muito menor, "
                   "porque quase não varia entre as alternativas. Com a normalização por faixa, a influência de "
                   "cada critério fica coerente com o peso declarado na matriz de Saaty.")

    # ------------------------------------------------------------------ 5. Sensibilidade
    with t_sens:
        st.markdown("#### Em que peso de custo a decisão muda?")
        st.caption("Varre o peso do custo de 0 a 100%; os demais critérios mantêm as proporções do AHP. "
                   "Usa a configuração escolhida acima.")
        grade, v_a, v_p = varredura_custo(M, w0, faixa_sel, p_sel, q_sel)
        st.plotly_chart(fig_intervalos(grade, v_a, v_p, float(w0[0])), width="stretch")
        ia = intervalos(grade, v_a)
        ip = intervalos(grade, v_p)
        st.dataframe(pd.DataFrame(
            [{"Método": nm, "Alternativa": rotulos[a], "Peso de custo de": f"{ini:.0%}", "até": f"{fim:.0%}"}
             for nm, lst in (("AHP", ia), ("PROMETHEE II", ip)) for ini, fim, a in lst]),
            hide_index=True, width="stretch")

        st.markdown("#### Pesos manuais")
        st.caption("Os sliders partem dos pesos do AHP e são renormalizados.")
        cols = st.columns(4)
        w_raw = [cols[k].slider(CURTO_CRIT[k], 0, 100, int(round(w0[k] * 100)), key=f"w{k}") for k in range(4)]
        w = np.array(w_raw, dtype=float)
        if w.sum() == 0:
            st.warning("Defina pelo menos um peso maior que zero.")
        else:
            w = w / w.sum()
            s_w = ahp_scores(M, w, faixa_sel)
            phi_w = promethee(M, w, p_sel, q_sel)
            k1, k2 = st.columns(2)
            k1.metric("Vencedora AHP (pesos manuais)", rotulos[int(np.argmax(s_w))])
            k2.metric("Vencedora PROMETHEE II (pesos manuais)", rotulos[int(np.argmax(phi_w))])
            st.dataframe(pd.DataFrame({"Score AHP": s_w, "Rank AHP": (-s_w).argsort().argsort() + 1,
                                       "Φ líquido": phi_w, "Rank PROMETHEE": (-phi_w).argsort().argsort() + 1},
                                      index=rotulos).style.format({"Score AHP": "{:.4f}", "Φ líquido": "{:+.4f}"}),
                         width="stretch")

        st.markdown("#### Robustez a variações dos pesos (Monte Carlo local, 2000 sorteios)")
        st.caption("Os pesos são sorteados **em torno dos pesos de Saaty** (Dirichlet centrada em w, sem perder a matriz): "
                   "concentração 30 ≈ ±9 p.p. no peso do custo; 100 ≈ ±5 p.p. Mostra a % de sorteios em que cada "
                   "alternativa vence, na configuração escolhida acima.")
        mc = {c: robustez_pesos(M, w0, faixa_sel, p_sel, q_sel, conc=c) for c in (30, 100)}
        mc_a, mc_p = mc[30]
        f = go.Figure()
        for c, dash in ((30, 1.0), (100, 0.55)):
            f.add_trace(go.Bar(x=rotulos, y=mc[c][0], name=f"AHP · conc. {c}", marker_color=RED, opacity=dash))
            f.add_trace(go.Bar(x=rotulos, y=mc[c][1], name=f"PROMETHEE · conc. {c}", marker_color=GOLD, opacity=dash))
        st.plotly_chart(layout_base(f, 360, barmode="group", yaxis_title="% de vitórias"), width="stretch")
        st.caption("Sortear pesos de uma Dirichlet uniforme (sem relação com Saaty) daria peso médio de custo de 25% e "
                   "contradiria o julgamento declarado; por isso não é usado.")

    # ------------------------------------------------------------------ 6. Solução
    with t_sol:
        escolha = st.selectbox("Alternativa a detalhar", rotulos, index=i_rec)
        i_sel = rotulos.index(escolha)
        if X_alts is None:
            st.info(f"Para detalhar a composição, deixe `{ARQ_SP}` e `{ARQ_EPS}` na pasta do app.")
        else:
            X = X_alts[i_sel]
            k = st.columns(5)
            k[0].metric("f1 · Custo (R$ mi)", mi(A.f1(X)))
            k[1].metric("f2 · SiO₂", br(A.f2(X), 3))
            k[2].metric("f3 · Al₂O₃", br(A.f3(X), 3))
            k[3].metric("f4 · Homogeneidade", br(A.f4_homogeneidade(X), 1))
            k[4].metric("Viável?", "Sim" if A.e_viavel(X) else "NÃO")

            tot2, vies2 = decomp_vies(X, D.sio2_m, D.alvo_sio2)
            tot3, vies3 = decomp_vies(X, D.al2o3_m, D.alvo_al2o3)
            st.caption(f"Viés × variação: {vies2:.0f}% do f2 e {vies3:.0f}% do f3 vêm do afastamento da **média** do equipamento "
                       "em relação ao alvo; o restante é variação entre pilhas do mesmo equipamento.")

            st.plotly_chart(fig_composicao(X), width="stretch")

            q_s, q_a = A.teor_pilha(X, D.sio2_m), A.teor_pilha(X, D.al2o3_m)
            tol = 1e-9
            tb = pd.DataFrame({
                "Pilha": [p["nome"] for p in D.PILHAS], "Sinter": [p["sinter"] for p in D.PILHAS],
                "Caminhões": X.sum(axis=0), "SiO₂ (%)": q_s, "Al₂O₃ (%)": q_a,
                "Dentro dos limites": ["✔" if (D.min_sio2[p] - tol <= q_s[p] <= D.max_sio2[p] + tol and
                                               D.min_al2o3[p] - tol <= q_a[p] <= D.max_al2o3[p] + tol) else "✘"
                                       for p in range(D.N_PILHAS)]})
            st.dataframe(tb.style.format({"SiO₂ (%)": "{:.3f}", "Al₂O₃ (%)": "{:.3f}"}), hide_index=True,
                         width="stretch")

            fs, fa = folga_operacional(X)
            st.caption(f"**Folga operacional** (menor distância de qualquer pilha ao limite mín/máx): SiO₂ {br(fs, 3)} p.p. · "
                       f"Al₂O₃ {br(fa, 3)} p.p. Folga pequena significa que um pequeno erro de teor do lote real já viola a especificação.")

            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(fig_qualidade(X, D.sio2_m, "SiO₂", D.alvo_sio2, D.min_sio2, D.max_sio2, BLUE),
                                width="stretch")
            with g2:
                st.plotly_chart(fig_qualidade(X, D.al2o3_m, "Al₂O₃", D.alvo_al2o3, D.min_al2o3, D.max_al2o3, ORANGE),
                                width="stretch")

            st.markdown("**Receitas por equipamento** — quanto mais parecidas as colunas, menor o f4.")
            h1, h2 = st.columns(2)
            with h1:
                st.plotly_chart(fig_heatmap_receitas(X, 1), width="stretch")
            with h2:
                st.plotly_chart(fig_heatmap_receitas(X, 2), width="stretch")

            uso = pd.DataFrame({"Usado": X.sum(axis=1), "Mín": D.dmin, "Máx": D.dmax}, index=D.NOMES_MINERIOS)
            with st.expander("Uso total por minério vs. disponibilidade"):
                st.dataframe(uso[uso["Usado"] > 0], width="stretch")

    # ------------------------------------------------------------------ 7. Decisão final e análise crítica
    with t_crit:
        # ---------- dados da decisão final
        r = rotulos[i_rec]
        concorda = (i_rec_a == i_rec_p)
        rho_rec = spearman((-sc_rec).argsort().argsort() + 1, (-phi_rec).argsort().argsort() + 1)
        mc30 = robustez_pesos(M, w0, faixa_ref_, p_rec, q_rec, conc=30)
        mc100 = robustez_pesos(M, w0, faixa_ref_, p_rec, q_rec, conc=100)
        fs_all = np.array([folga_operacional(x) for x in X_alts]) if X_alts is not None else None

        def _inicio_intervalo(lst, peso):
            for k_, (ini, fim, a) in enumerate(lst):
                if ini - 1e-9 <= peso <= fim + 1e-9:
                    return ini, fim, a, (lst[k_ - 1][2] if k_ > 0 else None)
            return None
        iv = _inicio_intervalo(ia_r, float(w0[0]))
        i_barata_ = int(np.argmin(M[:, 0]))
        # alternativa de "qualidade": a que vence quando o custo pesa pouco
        i_alt = iv[3] if iv and iv[3] is not None else int(np.argsort(-sc_rec)[1])

        st.markdown("### Decisão final")
        st.success(f"**Alternativa recomendada: {r}** — custo R$ {mi(M[i_rec, 0])} mi, f2 = {br(M[i_rec, 1], 3)}, "
                   f"f3 = {br(M[i_rec, 2], 3)}, f4 = {br(M[i_rec, 3], 1)}.")
        st.markdown(f"""
**Por que ela.** A decisão é coerente com o julgamento declarado na matriz de Saaty (custo = {br(w0[0] * 100, 1)}% do peso):
na configuração adotada (AHP normalizado pela faixa da Entrega 1 + PROMETHEE II com limiares proporcionais), os dois
métodos {'**convergem** para a ' + r if concorda else 'divergem'} (correlação de Spearman entre os rankings = {br(rho_rec, 2)}).
{'É também a alternativa de **menor custo** entre as ' + str(n_alt) + '.' if i_rec == i_barata_ else ''}

**Quão estável.** Sorteando os pesos em torno dos de Saaty, {r} vence em **{mc30[0][i_rec]:.0f}% (AHP) e {mc30[1][i_rec]:.0f}% (PROMETHEE)**
dos casos com variação de ±9 p.p. no peso do custo, e em **{mc100[0][i_rec]:.0f}% / {mc100[1][i_rec]:.0f}%** com ±5 p.p.""")
        if iv:
            st.markdown(f"No AHP, {r} vence enquanto o peso do custo for ≥ **{iv[0]:.0%}** (declarado: {br(w0[0] * 100, 1)}%). "
                        + (f"Abaixo disso, a vencedora passa a ser **{rotulos[iv[3]]}**." if iv[3] is not None else ""))

        st.markdown("**Ressalvas da recomendação**")
        ref_alt = i_alt
        d_ = M[i_rec] - M[ref_alt]
        st.markdown(f"- Em relação à {rotulos[ref_alt]} (a vencedora se o custo pesar menos): economiza **R$ {mi(abs(d_[0]))} mi "
                    f"({br(d_[0] / M[ref_alt, 0] * 100, 1)}%)**, mas f2 {d_[1] / M[ref_alt, 1] * 100:+.0f}%, "
                    f"f3 {d_[2] / M[ref_alt, 2] * 100:+.0f}% e f4 {d_[3] / M[ref_alt, 3] * 100:+.0f}% (piores).")
        if fs_all is not None:
            st.markdown(f"- **Folga operacional:** {r} tem folga mínima de {br(fs_all[i_rec, 1], 3)} p.p. no Al₂O₃ e "
                        f"{br(fs_all[i_rec, 0], 3)} p.p. no SiO₂ até os limites da especificação. É pequena: na prática, convém "
                        "recomendar uma margem de segurança nos lotes reais ou rodar o modelo com limites levemente mais apertados.")
        st.markdown(f"- A recomendação é **condicional ao peso do custo**: se a Cassotis valorizar qualidade acima do que a "
                    f"matriz declara, a {rotulos[ref_alt]} é a alternativa seguinte no ranking.")
        i_q = int(np.argmin(M[:, 2]))
        if i_q != i_rec and fs_all is not None:
            st.markdown(f"- Se a prioridade for **qualidade com folga operacional**, a melhor opção é a **{rotulos[i_q]}**: menor f3 "
                        f"({br(M[i_q, 2], 3)}) e f4 ({br(M[i_q, 3], 1)}), folga de Al₂O₃ de {br(fs_all[i_q, 1], 3)} p.p., ao custo de "
                        f"R$ {mi(M[i_q, 0] - M[i_rec, 0])} mi a mais que a {r}. Ela vence quando o custo pesa menos de ~{ia_r[0][1]:.0%}.")

        if fs_all is not None:
            st.markdown("**Folga operacional de cada alternativa**")
            tf = pd.DataFrame({"Custo (R$ mi)": [mi(v) for v in M[:, 0]], "Folga SiO₂ (p.p.)": fs_all[:, 0],
                               "Folga Al₂O₃ (p.p.)": fs_all[:, 1],
                               "Rank AHP": (-sc_rec).argsort().argsort() + 1,
                               "Rank PROMETHEE": (-phi_rec).argsort().argsort() + 1}, index=rotulos)
            st.dataframe(tf.style.format({"Folga SiO₂ (p.p.)": "{:.3f}", "Folga Al₂O₃ (p.p.)": "{:.3f}"}), width="stretch")

        # ---------- sensibilidade ao julgamento de Saaty
        st.markdown("#### A decisão depende do julgamento de Saaty?")
        st.caption("Varia só o julgamento *custo vs. SiO₂ e Al₂O₃* (a); os demais julgamentos ficam como declarados.")
        linhas_s = []
        for a_ in (1, 2, 3, 4, 5):
            Sa = np.array([[1, a_, a_, 5], [1 / a_, 1, 1, 3], [1 / a_, 1, 1, 3], [1 / 5, 1 / 3, 1 / 3, 1]], dtype=float)
            _, wa, cra = executar_ahp(M, Sa)
            linhas_s.append({"Julgamento a": f"{a_}" + (" (declarado)" if a_ == 3 else ""), "Peso do custo": f"{br(wa[0] * 100, 1)}%",
                             "CR": f"{cra:.3f}",
                             "AHP": rotulos[int(np.argmax(ahp_scores(M, wa, faixa_ref_)))],
                             "PROMETHEE (proporcional)": rotulos[int(np.argmax(promethee(M, wa, p_rec, q_rec)))],
                             "PROMETHEE (absolutos)": rotulos[int(np.argmax(promethee(M, wa, P_LIM, Q_LIM)))]})
        st.dataframe(pd.DataFrame(linhas_s), hide_index=True, width="stretch")
        st.caption("Julgamentos acima de 3 reforçam a mesma escolha; abaixo de 3 o custo perde a primazia e a decisão migra "
                   "para alternativas de melhor qualidade. Isso mostra onde a recomendação deixa de valer.")

        # ---------- análise crítica
        st.markdown("#### Análise crítica da configuração escolhida nos botões acima")
        todas = {}
        for modo in MODOS_AHP:
            todas[f"AHP · {MODOS_AHP[modo].split(' (')[0]}"] = int(np.argmax(ahp_scores(M, w0, faixa_modo(modo, M, M_glob, ref))))
        for modo in MODOS_PROM:
            p_, q_ = limiares(modo, faixa_ref_)
            todas[f"PROMETHEE · {MODOS_PROM[modo].split(' (')[0].split(' —')[0]}"] = int(np.argmax(promethee(M, w0, p_, q_)))
        contagem = pd.Series([rotulos[v] for v in todas.values()]).value_counts()

        st.markdown(f"- **Com a configuração escolhida:** AHP → **{rotulos[i_ahp]}**; PROMETHEE II → **{rotulos[i_prom]}**"
                    f"{' (convergem)' if i_ahp == i_prom else ' (**divergem**)'}.")
        st.markdown("- **Nas 5 combinações testadas** (3 normalizações do AHP + 2 versões de limiares do PROMETHEE): "
                    + "; ".join(f"{a} vence em {n}" for a, n in contagem.items()) + ".")
        st.markdown("- As combinações que divergem são as que **deixam de refletir o peso declarado**: a normalização pelo inverso "
                    "reduz o custo a ~7% de influência efetiva (aba 4), e os limiares absolutos toleram ~2× mais diferença de custo "
                    "(em % da faixa) do que nos demais critérios. Sob as configurações coerentes com os pesos, os métodos convergem.")

        st.markdown("**Pontos de virada (peso de custo) na configuração escolhida**")
        for nm, lst in (("AHP", ia), ("PROMETHEE II", ip)):
            st.markdown(f"- {nm}: " + "; ".join(f"**{rotulos[a]}** de {ini:.0%} a {fim:.0%}" for ini, fim, a in lst)
                        + f" (peso declarado: {w0[0]:.1%}).")

        st.markdown("- **Monte Carlo local (±9 p.p.):** "
                    + "; ".join(f"{rotulos[i]} vence {mc_a[i]:.0f}% (AHP) / {mc_p[i]:.0f}% (PROMETHEE)"
                                for i in np.argsort(-mc_a)[:3] if mc_a[i] > 0 or mc_p[i] > 0) + ".")

        # trade-off entre as duas candidatas
        a_, b_ = i_ahp, (i_prom if i_prom != i_ahp else int(np.argsort(-scores)[1]))
        st.markdown(f"#### Trade-off entre {rotulos[a_]} e {rotulos[b_]}")
        dif = M[a_] - M[b_]
        st.dataframe(pd.DataFrame({
            rotulos[a_]: [f"R$ {mi(M[a_, 0])} mi", br(M[a_, 1], 3), br(M[a_, 2], 3), br(M[a_, 3], 1)],
            rotulos[b_]: [f"R$ {mi(M[b_, 0])} mi", br(M[b_, 1], 3), br(M[b_, 2], 3), br(M[b_, 3], 1)],
            f"{rotulos[a_]} − {rotulos[b_]}": [f"{'+' if dif[0] >= 0 else '−'}R$ {mi(abs(dif[0]))} mi"] +
                                              [f"{'+' if d >= 0 else '−'}{br(abs(d), 3 if j < 2 else 1)}" for j, d in enumerate(dif[1:])],
            f"Variação % (base {rotulos[b_]})": [f"{dif[j] / M[b_, j] * 100:+.1f}%" for j in range(4)],
        }, index=NOMES_CRIT), width="stretch")
        barata, cara = (a_, b_) if M[a_, 0] <= M[b_, 0] else (b_, a_)
        d = M[cara] - M[barata]
        def _frase(j, nome, dec):
            v = (M[cara, j] - M[barata, j]) / M[barata, j] * 100
            return f"{nome} {'melhora' if v < 0 else 'piora'} {abs(v):.0f}% ({br(M[barata, j], dec)} → {br(M[cara, j], dec)})"
        st.markdown(f"Ir da **{rotulos[barata]}** (mais barata) para a **{rotulos[cara]}** custa **R$ {mi(d[0])} mi "
                    f"(+{d[0] / M[barata, 0] * 100:.1f}%)**; em troca, {_frase(1, 'f2', 3)}; {_frase(2, 'f3', 3)}; "
                    f"{_frase(3, 'f4', 1)} (percentuais sobre a {rotulos[barata]}, base diferente da coluna da tabela).")

        if X_alts is not None:
            vs2 = [decomp_vies(x, D.sio2_m, D.alvo_sio2)[1] for x in X_alts]
            vs3 = [decomp_vies(x, D.al2o3_m, D.alvo_al2o3)[1] for x in X_alts]
            texto_vies = (f"Nas {n_alt} alternativas, {min(vs2):.0f}–{max(vs2):.0f}% do f2 e {min(vs3):.0f}–{max(vs3):.0f}% do f3 "
                          "vêm do afastamento da **média** do equipamento ao alvo, e só o resto é variação entre pilhas — "
                          "que é o foco declarado do case (\"o vilão é a variação, não a média\"). Como as pilhas já ficam "
                          "quase iguais entre si, o ganho adicional de homogeneizá-las é pequeno; o custo do desvio está na média.")
        else:
            texto_vies = "Sem os .npz não é possível decompor o desvio em viés e variação."
        st.markdown("#### Limitações")
        st.markdown(f"""
1. **Gap das heurísticas.** O menor custo exato do problema é R$ {mi(LIMITES_EXATOS['f1'])} mi; o melhor f1 do VNS
   na Entrega 1 ficou ~3% acima disso, e os extremos de f2 e f3 também não são ótimos certificados (aba 1). A fronteira é
   uma aproximação; o VNS é estocástico e não prova otimalidade.
2. **Referências de normalização não são o ideal verdadeiro.** A Entrega 2 encontrou valores melhores que os extremos
   isolados da Entrega 1; por isso a normalização pode gerar valores levemente fora de [0, 1].
3. **Normalização, limiares e pesos são julgamentos.** Foram adotadas a normalização por faixa e os limiares proporcionais
   por serem coerentes com o peso declarado do custo. A normalização pelo inverso e os limiares absolutos reduzem a influência
   efetiva do custo e levam à Alt 5 (ver sensibilidade). A recomendação é condicional ao julgamento de Saaty (tabela acima).
4. **Os dois métodos não são independentes:** o PROMETHEE reutiliza os pesos do AHP, e a matriz de Saaty é um
   julgamento do grupo, sem validação externa. A concordância entre os métodos mostra que a escolha não é artefato da regra de agregação, mas não valida os pesos.
5. **f4 é medido em caminhões absolutos** e não entra na geração da fronteira (apenas na decisão); as pilhas têm
   tamanhos diferentes, então f4 = 0 é inatingível. Sua correlação com f2/f3 é fraca (não é redundante, mas o conflito
   é fraco), e com peso de 7,8% ele pouco influencia a escolha.
6. **Amostra de {n_alt} alternativas** (menor distância ao ponto utópico): favorece soluções de compromisso e deixa os
   extremos da fronteira de fora; a decisão é condicional a essa amostra.
7. **Soma ponderada não alcança regiões não convexas** da fronteira, e apenas parte das execuções do ε-restrito é viável
   em relação ao ε (veja a aba 1).
8. **Pesos de penalidade** (100/50) calibrados por tentativa; a sensibilidade a eles não foi quantificada.
9. **Média × variação.** {texto_vies}
""")