# =============================================================
# limites_exatos.py — Limites inferiores EXATOS de f1, f2 e f3 (scipy)
# Serve para medir o gap das heurísticas (VNS) das Entregas 1 e 2.
#   f1: MILP inteiro  -> ótimo exato do custo (com todas as restrições)
#   f2, f3: QP convexo na relaxação contínua -> limite inferior
#           (o ótimo inteiro é >= este valor; o gap calculado é, portanto, um teto)
# Uso:  python limites_exatos.py        (requer scipy)
# =============================================================
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, minimize

import dados as D

NI, NP = D.N_MINERIOS, D.N_PILHAS
n = D.n_cam_alvo.astype(float)
idx = lambda i, p: i * NP + p

# --- restrições lineares: massa, disponibilidade, qualidade (min/max do teor)
rows, lo, hi = [], [], []
for p in range(NP):
    r = np.zeros(NI * NP); r[[idx(i, p) for i in range(NI)]] = 1
    rows.append(r); lo.append(n[p]); hi.append(n[p])
for i in range(NI):
    r = np.zeros(NI * NP); r[[idx(i, p) for p in range(NP)]] = 1
    rows.append(r); lo.append(D.dmin[i]); hi.append(D.dmax[i])
for p in range(NP):
    for v, mn, mx in ((D.sio2_m, D.min_sio2[p], D.max_sio2[p]),
                      (D.al2o3_m, D.min_al2o3[p], D.max_al2o3[p])):
        r = np.zeros(NI * NP)
        for i in range(NI):
            r[idx(i, p)] = v[i]
        rows.append(r); lo.append(mn * n[p]); hi.append(mx * n[p])
A = np.array(rows); lo = np.array(lo); hi = np.array(hi)
ub = np.array([n[p] if D.elegivel[i, p] else 0.0 for i in range(NI) for p in range(NP)])

# --- f1 exato (MILP)
c = np.array([D.preco[i] * 2000 for i in range(NI) for p in range(NP)])
r1 = milp(c, constraints=LinearConstraint(A, lo, hi),
          bounds=Bounds(np.zeros(NI * NP), ub), integrality=np.ones(NI * NP))
print(f"f1 mínimo exato (MILP): R$ {r1.fun:,.2f}")

# --- f2 / f3: limite inferior contínuo (QP convexo)
def piso(v, alvo):
    def f(x):
        q = (v @ x.reshape(NI, NP)) / n
        return np.sum((q - alvo) ** 2)
    def g(x):
        q = (v @ x.reshape(NI, NP)) / n
        return ((2 * (q - alvo) / n)[None, :] * v[:, None]).ravel()
    res = minimize(f, r1.x, jac=g, method="SLSQP", bounds=list(zip(np.zeros(NI * NP), ub)),
                   constraints=[{"type": "ineq", "fun": lambda x: A @ x - lo, "jac": lambda x: A},
                                {"type": "ineq", "fun": lambda x: hi - A @ x, "jac": lambda x: -A}],
                   options={"maxiter": 500, "ftol": 1e-12})
    return res.fun

print(f"f2 limite inferior (relaxação contínua): {piso(D.sio2_m, D.alvo_sio2):.4f}")
print(f"f3 limite inferior (relaxação contínua): {piso(D.al2o3_m, D.alvo_al2o3):.4f}")
