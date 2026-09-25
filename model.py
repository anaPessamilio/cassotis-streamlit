import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from streamlit_pdf_viewer import pdf_viewer

@st.cache_data
def carregar_dados(filepath="data_piles.json"):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

data = carregar_dados()

# =========================================================================================================

## ------- STREAMLIT FRONT-END -------

st.set_page_config(layout="wide", page_title="Case Otimização UFMG")
st.logo(
    "side_bar_logo.png",
    link="https://www.cassotis.com/",
    icon_image="main_body_logo.png",
    size="large",
)
st.title("Case: Otimização de custo e qualidade na formação de pilhas de minério")
st.subheader("Cassotis Consulting - Engenharia de Sistemas UFMG 2026.2")
st.divider()

with st.sidebar:
    st.title("Menu Principal")

    aba_selecionada = st.radio(
        "Ir para:",
        ["Material auxiliar", "Dados de Entrada", "Decisões de Modelagem", "Resultados"]
    )
    
    st.divider()
    st.caption("Material desenvolvido e elaborado por Cassotis Consulting")



if aba_selecionada == "Material auxiliar":
    st.header("Material Auxiliar")
    st.markdown("""
    Apresentação do case:
    """)

    pdf_viewer("Case_UFMG2026.pdf")

elif aba_selecionada == "Dados de Entrada":

    ## -------------- Tabela de Minérios --------------
    st.header("Minérios")

    rows = []
    for m in data["Products"]:
        row = {
            "Minério": m,
            "Preço (R$/t)": data["m_product_price"][m],
            "FeT (%)": data["m_product_quality"][m]["FeT"],
            "SiO2 (%)": data["m_product_quality"][m]["SiO2"],
            "Al2O3 (%)": data["m_product_quality"][m]["Al2O3"],
            "Disp. mín (n° de caminhões)": data["m_product_delivery_availabity"][m]["min"],
            "Disp. máx (n° de caminhões)": data["m_product_delivery_availabity"][m]["max"],
            "Sinter 1": "☒" if data["m_product_equipment"][m]["Sinter 1"] else "☐",
            "Sinter 2": "☒" if data["m_product_equipment"][m]["Sinter 2"] else "☐",
        }
        rows.append(row)
    df_minerios = pd.DataFrame(rows).set_index("Minério")
    st.dataframe(df_minerios, use_container_width=True)

    ## -------------- Sinter 1 --------------
    st.header("Sinter 1")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Especificações de Qualidade")
        specs_s1 = data["m_equipment_quality_specs"]["Sinter 1"]
        spec_rows = []
        for label in ["min", "target", "max"]:
            display_label = {"min": "Mínimo", "target": "Alvo", "max": "Máximo"}[label]
            row = {"": display_label}
            for qi in data["QualityIndicators"]:
                val = specs_s1[qi][label]
                row[f"{qi} (%)"] = "N/A" if val is None else val
            spec_rows.append(row)
        df_specs_s1 = pd.DataFrame(spec_rows).set_index("")
        st.dataframe(df_specs_s1, use_container_width=True)

    with col2:
        st.subheader("Pilhas")
        pile_rows = []
        for p, mass in data["m_pile_weight"]["Sinter 1"].items():
            pile_rows.append({"Equipamento": "Sinter 1", "Pilha": p, "Massa da pilha (kt)": mass})
        df_piles_s1 = pd.DataFrame(pile_rows)
        st.dataframe(df_piles_s1, hide_index=True, use_container_width=True)

    ## -------------- Sinter 2 --------------
    st.header("Sinter 2")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Especificações de Qualidade")
        specs_s2 = data["m_equipment_quality_specs"]["Sinter 2"]
        spec_rows = []
        for label in ["min", "target", "max"]:
            display_label = {"min": "Mínimo", "target": "Alvo", "max": "Máximo"}[label]
            row = {"": display_label}
            for qi in data["QualityIndicators"]:
                val = specs_s2[qi][label]
                row[f"{qi} (%)"] = "N/A" if val is None else val
            spec_rows.append(row)
        df_specs_s2 = pd.DataFrame(spec_rows).set_index("")
        st.dataframe(df_specs_s2, use_container_width=True)

    with col2:
        st.subheader("Pilhas")
        pile_rows = []
        for p, mass in data["m_pile_weight"]["Sinter 2"].items():
            pile_rows.append({"Equipamento": "Sinter 2", "Pilha": p, "Massa da pilha (kt)": mass})
        df_piles_s2 = pd.DataFrame(pile_rows)
        st.dataframe(df_piles_s2, hide_index=True, use_container_width=True)

    ## -------------- Gráficos --------------
    st.header("Visualizações")

    BLUE = "#2a78d6"
    ORANGE = "#eb6834"
    AQUA = "#1baf7a"

    # 1. Preço por minério
    st.subheader("Preço por Minério")
    minerios = data["Products"]
    precos = [data["m_product_price"][m] for m in minerios]

    fig_preco = go.Figure(go.Bar(
        x=minerios, y=precos,
        marker_color=BLUE,
        marker_line_width=0,
        text=precos, textposition="outside", textfont_size=11,
    ))
    fig_preco.update_layout(
        yaxis_title="Preço (R$/t)",
        xaxis_title="Minério",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=40),
        height=380,
    )
    fig_preco.update_yaxes(gridcolor="#e5e5e5", gridwidth=1, zeroline=False)
    st.plotly_chart(fig_preco, use_container_width=True)

    # 2. Disponibilidade por minério (faixa min-max)
    st.subheader("Disponibilidade por Minério (nº de caminhões)")
    disp_min = [data["m_product_delivery_availabity"][m]["min"] for m in minerios]
    disp_max = [data["m_product_delivery_availabity"][m]["max"] for m in minerios]

    fig_disp = go.Figure()
    fig_disp.add_trace(go.Bar(
        x=minerios, y=disp_min,
        name="Mínimo",
        marker_color=BLUE, marker_line_width=0,
    ))
    fig_disp.add_trace(go.Bar(
        x=minerios, y=[mx - mn for mx, mn in zip(disp_max, disp_min)],
        name="Faixa disponível",
        marker_color=AQUA, marker_line_width=0,
        text=disp_max, textposition="outside", textfont_size=10,
    ))
    fig_disp.update_layout(
        barmode="stack",
        yaxis_title="Nº de caminhões",
        xaxis_title="Minério",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=40),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_disp.update_yaxes(gridcolor="#e5e5e5", gridwidth=1, zeroline=False)
    st.plotly_chart(fig_disp, use_container_width=True)

    # 3. Desvio em relação ao alvo — Sinter 1
    st.subheader("Desvio da Qualidade vs. Alvo — Sinter 1")
    specs_s1 = data["m_equipment_quality_specs"]["Sinter 1"]
    minerios_s1 = [m for m in minerios if data["m_product_equipment"][m]["Sinter 1"]]

    for qi, cor in [("SiO2", BLUE), ("Al2O3", ORANGE)]:
        alvo = specs_s1[qi]["target"]
        if alvo is None:
            continue
        desvios = [data["m_product_quality"][m][qi] - alvo for m in minerios_s1]
        cores = [AQUA if d < 0 else cor for d in desvios]
        fig_dev = go.Figure(go.Bar(
            x=minerios_s1, y=desvios,
            marker_color=cores, marker_line_width=0,
            text=[f"{d:+.2f}" for d in desvios], textposition="outside", textfont_size=10,
        ))
        fig_dev.add_hline(y=0, line_width=1.5, line_color="#333")
        fig_dev.update_layout(
            title=f"{qi} — Alvo Sinter 1: {alvo}%",
            yaxis_title=f"Desvio {qi} (p.p.)",
            xaxis_title="Minério",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=40),
            height=350,
        )
        fig_dev.update_yaxes(gridcolor="#e5e5e5", gridwidth=1, zeroline=False)
        st.plotly_chart(fig_dev, use_container_width=True)

    # 3. Desvio em relação ao alvo — Sinter 2
    st.subheader("Desvio da Qualidade vs. Alvo — Sinter 2")
    specs_s2 = data["m_equipment_quality_specs"]["Sinter 2"]
    minerios_s2 = [m for m in minerios if data["m_product_equipment"][m]["Sinter 2"]]

    for qi, cor in [("SiO2", BLUE), ("Al2O3", ORANGE)]:
        alvo = specs_s2[qi]["target"]
        if alvo is None:
            continue
        desvios = [data["m_product_quality"][m][qi] - alvo for m in minerios_s2]
        cores = [AQUA if d < 0 else cor for d in desvios]
        fig_dev = go.Figure(go.Bar(
            x=minerios_s2, y=desvios,
            marker_color=cores, marker_line_width=0,
            text=[f"{d:+.2f}" for d in desvios], textposition="outside", textfont_size=10,
        ))
        fig_dev.add_hline(y=0, line_width=1.5, line_color="#333")
        fig_dev.update_layout(
            title=f"{qi} — Alvo Sinter 2: {alvo}%",
            yaxis_title=f"Desvio {qi} (p.p.)",
            xaxis_title="Minério",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=40),
            height=350,
        )
        fig_dev.update_yaxes(gridcolor="#e5e5e5", gridwidth=1, zeroline=False)
        st.plotly_chart(fig_dev, use_container_width=True)

elif aba_selecionada == "Decisões de Modelagem":
    st.header("Decisões de Modelagem")
    st.markdown("""
    Nesta seção, descreva e justifique as principais decisões tomadas na construção do modelo e do algoritmo de solução.
    """)

    st.subheader("1. Codificação do Problema")
    st.info("Apresente a representação (codificação) adotada para a solução do problema — como um plano é descrito como estrutura de dados.")

    st.subheader("2. Estruturas de Vizinhança")
    st.info("Descreva as estruturas de vizinhança propostas (mínimo 3) e justifique a escolha de cada uma. Explique também como soluções inviáveis geradas pelos movimentos são tratadas.")

    st.subheader("3. Heurística Construtiva (Solução Inicial)")
    st.info("Explique a heurística utilizada para gerar a solução inicial e por que essa abordagem foi escolhida.")

    st.subheader("4. Métodos de Auxílio à Tomada de Decisão")
    st.info("Indique quais métodos multicritério foram escolhidos para apoiar a decisão e justifique a escolha.")

    st.subheader("5. Atributos de Interesse")
    st.info("Liste os atributos de interesse considerados na análise multicritério e justifique por que foram selecionados.")

elif aba_selecionada == "Resultados":
    st.header("Resultados do Modelo")
    st.markdown("""
    As seções abaixo são sugestões de estrutura para a apresentação dos resultados.
    **Vocês são incentivados a ir além** — acrescentem análises, visualizações e discussões adicionais que considerem relevantes.
    """)

    st.subheader("1. Soluções por Objetivo")
    st.info("Apresente os resultados obtidos para cada função objetivo (f₁: custo, f₂: desvio SiO₂, f₃: desvio Al₂O₃), otimizadas separadamente.")

    st.subheader("2. Composição das Pilhas")
    st.info("Mostre a alocação de minérios por pilha — quais minérios, quantos caminhões, e a qualidade resultante de cada pilha.")

    st.subheader("3. Qualidade Obtida vs. Especificações")
    st.info("Compare os teores de SiO₂ e Al₂O₃ obtidos em cada pilha com os limites mín/alvo/máx de cada equipamento.")

    st.subheader("4. Fronteira de Pareto")
    st.info("Apresente a fronteira de Pareto encontrada pela metaheurística, mostrando os trade-offs entre os objetivos.")

    st.subheader("5. Ranking Multicritério")
    st.info("Exiba o resultado do método de auxílio à tomada de decisão aplicado sobre as soluções da fronteira de Pareto.")