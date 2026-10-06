# -*- coding: utf-8 -*-
"""Verificacao do maximo do ZINB: hessiana (diferencas finitas do escore analitico)
e 400 perturbacoes aleatorias dos parametros (200 direcoes x 2 amplitudes)."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import json, numpy as np, warnings
warnings.filterwarnings("ignore")
import analise_tcc_final as A
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
R = json.load(open("resultados_finais.json"))
th = np.array([t["coef"] for t in R["zinb"]["tabela"]])
m = ZeroInflatedNegativeBinomialP(A.y, A.X, exog_infl=A.Xi, p=2, inflation="logit")
ll0 = m.loglike(th)
H = A.hessiana_por_escore(m, th); ev = np.linalg.eigvalsh(-H)
rng = np.random.default_rng(7); piores = 0; total = 0
for _ in range(200):
    d = rng.normal(size=len(th)); d /= np.linalg.norm(d)
    for s in (1e-3, 1e-2):
        total += 1; piores += int(m.loglike(th + s * d) < ll0)
out = {"llf": float(ll0), "autovalor_min_menos_H": float(ev.min()), "autovalor_max_menos_H": float(ev.max()),
       "perturbacoes": total, "perturbacoes_com_llf_menor": piores}
json.dump(out, open("verificacao_otimo.json", "w"), indent=1)
print(out)
