# -*- coding: utf-8 -*-
"""
reproduz_apendice.py
Executa os trechos de codigo do Apendice A do TCC, na ordem em que foram impressos,
e compara os resultados com resultados_finais.json (tempo aproximado: 10 min).
"""
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomialP
from statsmodels.discrete.count_model import (ZeroInflatedPoisson,
    ZeroInflatedNegativeBinomialP)

# 1) Especificação: contagem (intensidade) e inflação (ausência estrutural);
#    acordo_sanitario = histórico de exportação ao parceiro em 2000-2013
GRAV = ["ln_pib_parc", "ln_dist", "rta", "idioma_comum", "cambio", "preco_lag"]
INFL = ["ln_pib_pc_parc", "acordo_sanitario", "ln_dist", "idioma_comum", "rta"]
df = pd.read_csv("painel_real_milhares.csv")
dc = pd.get_dummies(df["commodity"], drop_first=True, dtype=float)
da = pd.get_dummies(df["ano"], drop_first=True, dtype=float)
X = sm.add_constant(pd.concat([df[GRAV].astype(float), dc, da], axis=1))
Xi = sm.add_constant(df[INFL].astype(float))
y = df["valor_exportado"].astype(float)

# 2) Poisson por IRLS (= PPML) e teste de Cameron e Trivedi (1990)
poi = sm.GLM(y, X, family=sm.families.Poisson()).fit(tol=1e-12)
mu = poi.fittedvalues
ct = sm.OLS(((y - mu)**2 - y) / mu, mu).fit()  # t > 0: superdispersão

# 3) Binomial Negativa (variância quadrática): alfa inicial pelo método dos momentos
a0 = max((np.var(y) - y.mean()) / y.mean()**2, 0.5)
nbm = NegativeBinomialP(y, X, p=2)
nb = nbm.fit(start_params=np.r_[poi.params, a0], method="bfgs", maxiter=5000)
nb = nbm.fit(start_params=nb.params, method="newton")  # refinamento

# 4) ZINB: protocolo de múltiplos pontos de partida (50 estimações)
lg = sm.Logit((y == 0).astype(float), Xi).fit(disp=0)
zp = ZeroInflatedPoisson(y, X, exog_infl=Xi, inflation="logit").fit(
    start_params=np.r_[lg.params, poi.params], method="bfgs", maxiter=5000)
b, a = nb.params.values[:-1], nb.params.values[-1]
partidas = [None,  # valor-padrão da biblioteca
    np.r_[[-3, 0, 0, 0, 0, 0], b, a],  # a partir da NB
    np.r_[lg.params, b, a],  # logit dos zeros e NB
    np.r_[lg.params, poi.params, a0],  # logit, Poisson e alfa inicial
    np.r_[zp.params, a]]  # ZIP e alfa da NB
rng = np.random.default_rng(2026)  # 45 partidas aleatórias
for _ in range(45):
    g = lg.params.values + rng.normal(0, 1.5, 6)
    bb = b.copy(); bb[0] += rng.normal(0, 0.5)
    bb[1:7] += rng.normal(0, 0.3, 6); bb[7:] += rng.normal(0, 0.2, len(bb) - 7)
    partidas.append(np.r_[g, bb, np.exp(rng.uniform(0, np.log(30)))])
zm = ZeroInflatedNegativeBinomialP(y, X, exog_infl=Xi, p=2, inflation="logit")
ajustes = [zm.fit(start_params=sp, method="bfgs", maxiter=6000, disp=0,
    skip_hessian=True) for sp in partidas]
melhor = max(ajustes, key=lambda r: r.llf)  # maior log-verossimilhança

# 5) Hessiana por diferenças finitas do escore e erros-padrão por agrupamento
th = melhor.params.values; k = len(th); H = np.zeros((k, k))
for j in range(k):
    h = 1e-5 * max(1.0, abs(th[j])); e = np.zeros(k); e[j] = h
    H[:, j] = (zm.score(th + e) - zm.score(th - e)) / (2 * h)
H = (H + H.T) / 2  # np.linalg.eigvalsh(-H) > 0: máximo local
V = np.linalg.inv(-H); S = zm.score_obs(th)
par = (df["pais"] + "_" + df["commodity"]).values
gr, idx = np.unique(par, return_inverse=True)
Sg = np.zeros((len(gr), k)); np.add.at(Sg, idx, S)
G, n = len(gr), len(y)
V_cl = G / (G - 1) * (n - 1) / (n - k) * V @ Sg.T @ Sg @ V
ep_cl = np.sqrt(np.diag(V_cl))

# 6) Tipologia de parceiros: escolha de K pela inércia e pela silhueta média
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
f = df.groupby("pais")[["ln_pib_parc", "ln_pib_pc_parc", "ln_dist"]].mean()
Zf = StandardScaler().fit_transform(f)
km, inercia, silhueta = {}, {}, {}
for K in range(2, 9):  # partições de dois a oito grupos
    km[K] = KMeans(n_clusters=K, n_init=50, random_state=42).fit(Zf)
    inercia[K] = km[K].inertia_  # soma dos quadrados intragrupos
    silhueta[K] = silhouette_score(Zf, km[K].labels_)  # silhueta média
tipo = pd.Series(km[4].labels_, index=f.index)  # partição adotada: K = 4

# 7) Margem de entrada: GLMM logístico com intercepto aleatório por parceiro,
#    por máxima verossimilhança com quadratura de Gauss-Hermite adaptativa
#    (rotina própria; otimização pela biblioteca SciPy)
from scipy import optimize, special
from numpy.polynomial.hermite import hermgauss

def hess_num(f, th, h=1e-4):  # hessiana por diferenças finitas centrais
    k = len(th); E = np.eye(k) * h; H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            H[i, j] = H[j, i] = (f(th + E[i] + E[j]) - f(th + E[i] - E[j])
                - f(th - E[i] + E[j]) + f(th - E[i] - E[j])) / (4 * h * h)
    return H

class GLMMLogitAGQ:
    def __init__(self, y, X, grupos, nq=15):
        self.y, self.X = np.asarray(y, float), np.asarray(X, float)
        self.g, self.gi = np.unique(grupos, return_inverse=True)
        self.G, self.nq = len(self.g), nq
        self.t, self.w = hermgauss(nq)  # nós e pesos de Gauss-Hermite
    def modos(self, eta, s):  # modo e escala do integrando de cada parceiro
        u = np.zeros(self.G)
        for _ in range(200):  # Newton amortecido (integrando log-côncavo)
            p = special.expit(eta + u[self.gi])
            gr = np.bincount(self.gi, self.y - p, self.G) - u / s**2
            he = -np.bincount(self.gi, p * (1 - p), self.G) - 1 / s**2
            passo = np.clip(gr / he, -2, 2); u = u - passo
            if np.abs(passo).max() < 1e-12: break
        p = special.expit(eta + u[self.gi])
        c = np.bincount(self.gi, p * (1 - p), self.G) + 1 / s**2
        return u, 1 / np.sqrt(c)
    def loglik(self, th):  # th: coeficientes e log do desvio-padrão de u
        s = np.exp(th[-1]); eta = self.X @ th[:-1]
        u, tau = self.modos(eta, s)
        nos = u[:, None] + np.sqrt(2) * tau[:, None] * self.t  # nós adaptados
        z = eta[:, None] + nos[self.gi]
        lf = self.y[:, None] * z - np.logaddexp(0, z)  # log f(y | u)
        S = np.zeros((self.G, self.nq)); np.add.at(S, self.gi, lf)
        lp = -0.5 * np.log(2 * np.pi * s**2) - nos**2 / (2 * s**2)
        li = S + lp + self.t**2 + np.log(self.w)
        lg = np.log(np.sqrt(2) * tau) + special.logsumexp(li, axis=1)
        return np.sum(lg)  # soma das log-verossimilhanças dos parceiros
    def fit(self, inicio):
        f = lambda th: -self.loglik(th)
        op = {"gtol": 1e-5, "maxiter": 2000}
        bfgs = lambda x0: optimize.minimize(f, x0, method="BFGS",
                                            jac="3-point", options=op)
        r = bfgs(inicio)
        if not r.success:  # alternativa: Nelder-Mead seguido de novo BFGS
            nm = optimize.minimize(f, r.x, method="Nelder-Mead", options={
                "maxiter": 20000, "xatol": 1e-7, "fatol": 1e-8})
            r = min([r, nm, bfgs(nm.x)], key=lambda z: z.fun)
        self.theta, self.llf = r.x, -r.fun
        self.ep = np.sqrt(np.diag(np.linalg.inv(hess_num(f, r.x))))
        return self

FULL = ["ln_pib_parc", "ln_pib_pc_parc", "ln_dist", "rta", "idioma_comum",
        "acordo_sanitario", "cambio", "preco_lag"]
yb = (y > 0).astype(float).values  # ocorrência de exportação
Xg = sm.add_constant(pd.concat([df[FULL].astype(float), dc, da], axis=1))
g0 = GLMMLogitAGQ(yb, np.ones((len(yb), 1)), df["pais"])  # modelo nulo
g0.fit(np.r_[0.0, 0.5])
lgb = sm.Logit(yb, Xg).fit(disp=0, method="bfgs", maxiter=3000)
ini = np.r_[lgb.params, g0.theta[-1]]  # logit simples e variância do nulo
glmm = GLMMLogitAGQ(yb, Xg, df["pais"]).fit(ini)  # modelo completo
s2u = np.exp(2 * glmm.theta[-1]); icc_g = s2u / (s2u + np.pi**2 / 3)

# 8) Margem de intensidade: HLM2 com intercepto aleatório por parceiro
from statsmodels.regression.mixed_linear_model import MixedLM
pos = df[y > 0].assign(ln_y=lambda d: np.log(d["valor_exportado"]))
hlm0 = MixedLM(pos["ln_y"], np.ones(len(pos)), groups=pos["pais"]).fit(
    reml=False, method="bfgs")
tau2 = np.asarray(hlm0.cov_re)[0, 0]
icc = tau2 / (tau2 + hlm0.scale)  # correlação intraclasse
Xh = sm.add_constant(pd.concat([pos[FULL].astype(float),
    pd.get_dummies(pos["commodity"], drop_first=True, dtype=float),
    pd.get_dummies(pos["ano"], drop_first=True, dtype=float)], axis=1))
hlm = MixedLM(pos["ln_y"], Xh, groups=pos["pais"]).fit(reml=False,
    method="bfgs", maxiter=5000)

# ---------------------------------------------------------------- verificacao (nao faz parte do Apendice A)
import json
from scipy import stats
R = json.load(open("resultados_finais.json"))
print("NB    llf", round(nb.llf, 4), "| ref", round(R["nb"]["llf"], 4))
print("ZIP   llf", round(zp.llf, 1), "| ref", round(R["zip"]["llf"], 1))
lls = sorted([r.llf for r in ajustes], reverse=True)
print("ZINB  llf", round(melhor.llf, 4), "| ref", round(R["zinb"]["llf"], 4))
print("partidas no otimo:", sum(abs(l - R["zinb"]["llf"]) < 1 for l in lls),
      "| colapsos na NB:", sum(abs(l - R["nb"]["llf"]) < 1 for l in lls))
print("menor autovalor de -H:", round(np.linalg.eigvalsh(-H).min(), 4))
print("silhueta (K = 2 a 8):", [round(silhueta[K], 3) for K in range(2, 9)])
print("tamanho dos grupos (K = 4):", np.bincount(tipo.values))
gC = R["glmm"]["completo"]
for v in ["ln_pib_parc", "ln_dist", "rta", "acordo_sanitario"]:
    i = list(Xg.columns).index(v)
    p = 2 * stats.norm.sf(abs(glmm.theta[i] / glmm.ep[i]))
    print(f"GLMM {v:16s} {glmm.theta[i]: .3f} (p = {p:.3f}) | ref {gC['coef'][v]['b']: .3f}")
print("GLMM ICC (nulo, completo):", round(np.exp(2 * g0.theta[-1]) / (np.exp(2 * g0.theta[-1]) + np.pi**2 / 3), 3),
      round(icc_g, 3), "| ref", round(R["glmm"]["nulo"]["icc"], 3), round(gC["icc"], 3))
print("HLM2 ICC nulo:", round(icc, 3), "| ref", round(R["hlm"]["nulo"]["icc"], 3))
print("HLM2 distancia:", round(hlm.fe_params["ln_dist"], 3), "(p =", round(hlm.pvalues["ln_dist"], 3), ")",
      "| ref", round(R["hlm"]["completo"]["coef"]["ln_dist"]["b"], 3))
