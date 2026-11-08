# =============================================================
# dados.py — Dados do problema (Anexo do enunciado Cassotis)
# =============================================================

import numpy as np

# ------------------------------------------------------------------
# MINÉRIOS: índice 0..19 → M1..M20
# Campos: preco (R$/t), FeT (%), SiO2 (%), Al2O3 (%),
#         disp_min (caminhões), disp_max (caminhões),
#         sinter1 (bool), sinter2 (bool)
# ------------------------------------------------------------------
MINERIOS = {
    #  id   preco  FeT   SiO2  Al2O3  dmin  dmax   S1     S2
    "M1":  {"preco": 320, "FeT": 66, "SiO2": 5.0, "Al2O3": 1.0, "dmin": 0, "dmax": 6,  "S1": True,  "S2": True},
    "M2":  {"preco": 280, "FeT": 64, "SiO2": 3.0, "Al2O3": 3.0, "dmin": 0, "dmax": 8,  "S1": True,  "S2": True},
    "M3":  {"preco": 270, "FeT": 62, "SiO2": 9.0, "Al2O3": 0.6, "dmin": 0, "dmax": 8,  "S1": True,  "S2": True},
    "M4":  {"preco": 190, "FeT": 58, "SiO2": 8.0, "Al2O3": 2.8, "dmin": 2, "dmax": 12, "S1": True,  "S2": True},
    "M5":  {"preco": 230, "FeT": 60, "SiO2": 6.0, "Al2O3": 1.5, "dmin": 0, "dmax": 10, "S1": True,  "S2": True},
    "M6":  {"preco": 300, "FeT": 63, "SiO2": 5.5, "Al2O3": 1.2, "dmin": 0, "dmax": 8,  "S1": True,  "S2": True},
    "M7":  {"preco": 260, "FeT": 65, "SiO2": 5.5, "Al2O3": 1.1, "dmin": 0, "dmax": 8,  "S1": True,  "S2": True},
    "M8":  {"preco": 240, "FeT": 61, "SiO2": 6.2, "Al2O3": 1.6, "dmin": 0, "dmax": 8,  "S1": True,  "S2": False},
    "M9":  {"preco": 210, "FeT": 59, "SiO2": 6.5, "Al2O3": 1.8, "dmin": 0, "dmax": 8,  "S1": True,  "S2": False},
    "M10": {"preco": 180, "FeT": 57, "SiO2": 7.0, "Al2O3": 2.5, "dmin": 2, "dmax": 10, "S1": True,  "S2": False},
    "M11": {"preco": 310, "FeT": 66, "SiO2": 4.7, "Al2O3": 0.7, "dmin": 1, "dmax": 6,  "S1": True,  "S2": False},
    "M12": {"preco": 200, "FeT": 60, "SiO2": 5.8, "Al2O3": 1.4, "dmin": 0, "dmax": 8,  "S1": True,  "S2": False},
    "M13": {"preco": 160, "FeT": 56, "SiO2": 9.0, "Al2O3": 3.2, "dmin": 0, "dmax": 10, "S1": True,  "S2": True},
    "M14": {"preco": 150, "FeT": 56, "SiO2": 6.0, "Al2O3": 1.3, "dmin": 0, "dmax": 6,  "S1": True,  "S2": True},
    "M15": {"preco": 170, "FeT": 58, "SiO2": 8.0, "Al2O3": 3.5, "dmin": 0, "dmax": 8,  "S1": True,  "S2": True},
    "M16": {"preco": 220, "FeT": 59, "SiO2": 6.0, "Al2O3": 1.5, "dmin": 0, "dmax": 8,  "S1": False, "S2": True},
    "M17": {"preco": 140, "FeT": 56, "SiO2": 8.5, "Al2O3": 3.6, "dmin": 0, "dmax": 10, "S1": False, "S2": True},
    "M18": {"preco": 260, "FeT": 62, "SiO2": 5.0, "Al2O3": 1.0, "dmin": 0, "dmax": 8,  "S1": False, "S2": True},
    "M19": {"preco": 190, "FeT": 57, "SiO2": 6.3, "Al2O3": 1.4, "dmin": 0, "dmax": 8,  "S1": False, "S2": True},
    "M20": {"preco": 130, "FeT": 56, "SiO2": 3.8, "Al2O3": 3.8, "dmin": 1, "dmax": 8,  "S1": False, "S2": True},
}

NOMES_MINERIOS = list(MINERIOS.keys())   # ["M1", ..., "M20"]
N_MINERIOS = len(NOMES_MINERIOS)         # 20

# ------------------------------------------------------------------
# PILHAS
# Pilhas 0..4 → Sinter 1 (índices internos)
# Pilhas 5..9 → Sinter 2
# massa_kt: massa-alvo em kt
# n_caminhoes: massa / 2 kt (número exato de caminhões)
# ------------------------------------------------------------------
PILHAS = [
    {"nome": "Pilha 1",  "sinter": 1, "massa_kt": 20, "n_cam": 10},
    {"nome": "Pilha 2",  "sinter": 1, "massa_kt": 22, "n_cam": 11},
    {"nome": "Pilha 3",  "sinter": 1, "massa_kt": 24, "n_cam": 12},
    {"nome": "Pilha 4",  "sinter": 1, "massa_kt": 26, "n_cam": 13},
    {"nome": "Pilha 5",  "sinter": 1, "massa_kt": 28, "n_cam": 14},
    {"nome": "Pilha 6",  "sinter": 2, "massa_kt": 30, "n_cam": 15},
    {"nome": "Pilha 7",  "sinter": 2, "massa_kt": 32, "n_cam": 16},
    {"nome": "Pilha 8",  "sinter": 2, "massa_kt": 34, "n_cam": 17},
    {"nome": "Pilha 9",  "sinter": 2, "massa_kt": 36, "n_cam": 18},
    {"nome": "Pilha 10", "sinter": 2, "massa_kt": 38, "n_cam": 19},
]
N_PILHAS = len(PILHAS)   # 10

# Índices das pilhas por sinter
IDX_S1 = [p for p, info in enumerate(PILHAS) if info["sinter"] == 1]  # [0,1,2,3,4]
IDX_S2 = [p for p, info in enumerate(PILHAS) if info["sinter"] == 2]  # [5,6,7,8,9]

# ------------------------------------------------------------------
# ESPECIFICAÇÕES DE QUALIDADE POR SINTER
# ------------------------------------------------------------------
SPECS = {
    1: {
        "SiO2_min":  5.0,  "SiO2_alvo": 5.55, "SiO2_max":  6.2,
        "Al2O3_min": 1.0,  "Al2O3_alvo":1.25, "Al2O3_max": 2.3,
    },
    2: {
        "SiO2_min":  5.0,  "SiO2_alvo": 5.50, "SiO2_max":  6.5,
        "Al2O3_min": 1.0,  "Al2O3_alvo":1.5,  "Al2O3_max": 2.2,
    },
}

# ------------------------------------------------------------------
# MATRIZES PRÉ-COMPUTADAS (para uso no VNS)
# ------------------------------------------------------------------

# preco[i]: preço do minério i (R$/t)
preco  = np.array([MINERIOS[m]["preco"]  for m in NOMES_MINERIOS], dtype=float)
sio2_m = np.array([MINERIOS[m]["SiO2"]  for m in NOMES_MINERIOS], dtype=float)
al2o3_m= np.array([MINERIOS[m]["Al2O3"] for m in NOMES_MINERIOS], dtype=float)
dmin   = np.array([MINERIOS[m]["dmin"]  for m in NOMES_MINERIOS], dtype=int)
dmax   = np.array([MINERIOS[m]["dmax"]  for m in NOMES_MINERIOS], dtype=int)

# elegivel[i, p] = True se minério i pode ir para pilha p
elegivel = np.zeros((N_MINERIOS, N_PILHAS), dtype=bool)
for i, m in enumerate(NOMES_MINERIOS):
    for p, pilha in enumerate(PILHAS):
        s = pilha["sinter"]
        elegivel[i, p] = MINERIOS[m][f"S{s}"]

# n_cam_alvo[p]: número de caminhões-alvo da pilha p
n_cam_alvo = np.array([pilha["n_cam"] for pilha in PILHAS], dtype=int)

# sinter_da_pilha[p]: 1 ou 2
sinter_da_pilha = np.array([pilha["sinter"] for pilha in PILHAS], dtype=int)

# alvo_sio2[p], alvo_al2o3[p], lim_sio2[p], lim_al2o3[p]
alvo_sio2  = np.array([SPECS[pilha["sinter"]]["SiO2_alvo"]  for pilha in PILHAS])
alvo_al2o3 = np.array([SPECS[pilha["sinter"]]["Al2O3_alvo"] for pilha in PILHAS])
min_sio2   = np.array([SPECS[pilha["sinter"]]["SiO2_min"]   for pilha in PILHAS])
max_sio2   = np.array([SPECS[pilha["sinter"]]["SiO2_max"]   for pilha in PILHAS])
min_al2o3  = np.array([SPECS[pilha["sinter"]]["Al2O3_min"]  for pilha in PILHAS])
max_al2o3  = np.array([SPECS[pilha["sinter"]]["Al2O3_max"]  for pilha in PILHAS])
