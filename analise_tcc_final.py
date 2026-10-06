# -*- coding: utf-8 -*-
"""
analise_tcc_final.py
Modelo gravitacional com inflacao de zeros para as exportacoes agropecuarias
brasileiras (2014-2024). Rotina final e reprodutivel do TCC (MBA DSA USP/Esalq).

Etapas: (1) descritivas; (2) Poisson (IRLS) e diagnosticos de dispersao;
(3) Binomial Negativa (NB2); (4) ZIP; (5) ZINB com protocolo de multiplos
pontos de partida; (6) PPML; (7) sensibilidade a unidade de medida;
(8) tipologia de parceiros (K-Means) e modelos multinivel (GLMM e HLM2).
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json, time, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats
warnings.filterwarnings("ignore")
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Poisson, NegativeBinomialP
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP
from statsmodels.regression.mixed_linear_model import MixedLM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from glmm_agq import GLMMLogitAGQ

GRAV = ["ln_pib_parc", "ln_dist", "rta", "idioma_comum", "cambio", "preco_lag"]   # contagem
INFL = ["ln_pib_pc_parc", "acordo_sanitario", "ln_dist", "idioma_comum", "rta"]   # inflacao
SEED = 2026
R = {}                                                                             # resultados

def carregar(arq):
    df = pd.read_csv(arq)
    dc = pd.get_dummies(df["commodity"], prefix="comm", drop_first=True, dtype=float)
    da = pd.get_dummies(df["ano"], prefix="ano", drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([df[GRAV].astype(float), dc, da], axis=1))
    Xi = sm.add_constant(df[INFL].astype(float))
    return df, df["valor_exportado"].astype(float), X, Xi

def alpha_momentos(y):
    m, v = float(np.mean(y)), float(np.var(y))
    return max((v - m) / m ** 2, 0.5)

def hessiana_por_escore(modelo, th, rel=1e-5):
    """Hessiana por diferencas finitas centrais do escore analitico."""
    k = len(th); H = np.zeros((k, k))
    for j in range(k):
        h = rel * max(1.0, abs(th[j])); e = np.zeros(k); e[j] = h
        H[:, j] = (modelo.score(th + e) - modelo.score(th - e)) / (2 * h)
    return 0.5 * (H + H.T)

def cov_cluster(S, cov, grupos):
    """Matriz de covariancia sanduiche robusta a agrupamentos (clusters)."""
    g, idx = np.unique(grupos, return_inverse=True); G = len(g); n, k = S.shape
    Sg = np.zeros((G, k)); np.add.at(Sg, idx, S)
    return (G / (G - 1)) * ((n - 1) / (n - k)) * cov @ (Sg.T @ Sg) @ cov, G

def tabela(nomes, th, cov, cov_par, cov_pais):
    pv = lambda b, s: float(2 * stats.norm.sf(abs(b / s)))
    out = []
    for i, nm in enumerate(nomes):
        se, sp, sc = [float(np.sqrt(c[i, i])) for c in (cov, cov_par, cov_pais)]
        out.append({"var": nm, "coef": float(th[i]), "se": se, "p": pv(th[i], se), "se_par": sp,
                    "p_par": pv(th[i], sp), "se_pais": sc, "p_pais": pv(th[i], sc)})
    return out

# ----------------------------------------------------------------------------
df, y, X, Xi = carregar("painel_real_milhares.csv")
n = len(y); par = (df["pais"] + "_" + df["commodity"]).values; pais = df["pais"].values

def ajusta_zinb(sp):
    m = ZeroInflatedNegativeBinomialP(y, X, exog_infl=Xi, p=2, inflation="logit")
    t0 = time.time()
    try:
        kw = dict(method="bfgs", maxiter=6000, disp=0, skip_hessian=True)
        if sp is not None: kw["start_params"] = sp
        r = m.fit(**kw)
        return {"llf": float(r.llf), "params": r.params.values.tolist(), "t": time.time() - t0}
    except Exception as e:
        return {"llf": -np.inf, "erro": str(e)}

if __name__ == "__main__":
    T0 = time.time()
    # ------------------------------------------------------------ (1) descritivas
    yv = y.values; pos = yv[yv > 0]
    R["eda"] = {"n": n, "n_paises": int(df["pais"].nunique()), "n_commodities": int(df["commodity"].nunique()),
                "anos": [int(df["ano"].min()), int(df["ano"].max())], "n_zeros": int((yv == 0).sum()),
                "pct_zeros": float((yv == 0).mean()), "n_pos": int((yv > 0).sum()), "media": float(yv.mean()),
                "variancia": float(yv.var()), "razao_var_media": float(yv.var() / yv.mean()),
                "cv": float(yv.std() / yv.mean()), "mediana_pos": float(np.median(pos)),
                "p25_pos": float(np.percentile(pos, 25)), "p75_pos": float(np.percentile(pos, 75)),
                "max": float(yv.max()), "total": float(yv.sum()),
                "paises_com_fluxo": int(df.loc[df["valor_exportado"] > 0, "pais"].nunique()),
                "obs_por_pais_min": int(df.groupby("pais").size().min())}
    com = df.groupby("commodity")["valor_exportado"].agg(
        n="size", pct_zeros=lambda s: float((s == 0).mean()), media="mean", total="sum").reset_index()
    R["eda_commodity"] = com.sort_values("total", ascending=False).to_dict("records")
    R["eda_ano"] = df.groupby("ano")["valor_exportado"].sum().to_dict()
    # ------------------------------------------------------------ (2) Poisson
    rp = Poisson(y, X).fit(method="bfgs", maxiter=1000, disp=0)
    glm = sm.GLM(y, X, family=sm.families.Poisson()).fit(maxiter=300, tol=1e-12)
    mu = glm.fittedvalues.values
    aux = ((yv - mu) ** 2 - yv) / mu
    ct, ct_r = sm.OLS(aux, mu).fit(), sm.OLS(aux, mu).fit(cov_type="HC1")
    R["poisson"] = {"llf_bfgs": float(rp.llf), "conv_bfgs": bool(rp.mle_retvals.get("converged")),
                    "grad_max_bfgs": float(np.max(np.abs(rp.model.score(rp.params.values)))),
                    "llf_irls": float(glm.llf), "iter_irls": int(glm.fit_history["iteration"]), "k": X.shape[1],
                    "aic": float(-2 * glm.llf + 2 * X.shape[1]), "bic": float(-2 * glm.llf + np.log(n) * X.shape[1]),
                    "zeros_prev": float(np.exp(-mu).sum()),
                    "pearson": float(np.sum((yv - mu) ** 2 / mu) / (n - X.shape[1])),
                    "ct_alpha": float(ct.params[0]), "ct_t": float(ct.tvalues[0]), "ct_p": float(ct.pvalues[0]),
                    "ct_t_hc1": float(ct_r.tvalues[0]), "ct_p_hc1": float(ct_r.pvalues[0])}
    print("Poisson ok", R["poisson"]["llf_irls"], flush=True)
    # ------------------------------------------------------------ (3) NB2
    a0 = alpha_momentos(yv); R["alpha0_momentos"] = a0
    nbm = NegativeBinomialP(y, X, p=2)
    nb = nbm.fit(start_params=np.r_[glm.params.values, a0], method="bfgs", maxiter=5000, disp=0)
    nb = nbm.fit(start_params=nb.params.values, method="newton", maxiter=100, disp=0)
    th_nb = nb.params.values; alpha_nb = float(th_nb[-1]); mu_nb = np.exp(X.values @ th_nb[:-1])
    cov_nb = np.linalg.inv(-nbm.hessian(th_nb)); S_nb = nbm.score_obs(th_nb)
    nb_null = NegativeBinomialP(y, np.ones((n, 1)), p=2).fit(disp=0, maxiter=2000)
    R["nb"] = {"llf": float(nb.llf), "k": len(th_nb), "aic": float(nb.aic), "bic": float(nb.bic), "alpha": alpha_nb,
               "llf_nulo": float(nb_null.llf), "mcfadden": float(1 - nb.llf / nb_null.llf),
               "zeros_prev": float(((1 + alpha_nb * mu_nb) ** (-1 / alpha_nb)).sum()),
               "pearson": float(np.sum((yv - mu_nb) ** 2 / (mu_nb + alpha_nb * mu_nb ** 2)) / (n - len(th_nb))),
               "grad_max": float(np.max(np.abs(nbm.score(th_nb)))),
               "lr_vs_poisson": float(2 * (nb.llf - glm.llf)),
               "tabela": tabela(list(X.columns) + ["alpha"], th_nb, cov_nb,
                                cov_cluster(S_nb, cov_nb, par)[0], cov_cluster(S_nb, cov_nb, pais)[0])}
    print("NB ok", R["nb"]["llf"], flush=True)
    # ------------------------------------------------------------ (4) ZIP
    lg = sm.Logit((y == 0).astype(float), Xi).fit(disp=0, method="bfgs", maxiter=2000)
    zm = ZeroInflatedPoisson(y, X, exog_infl=Xi, inflation="logit")
    zips = [zm.fit(method="bfgs", maxiter=5000, disp=0, skip_hessian=True),
            zm.fit(start_params=np.r_[lg.params.values, glm.params.values], method="bfgs", maxiter=5000, disp=0, skip_hessian=True)]
    zp = max(zips, key=lambda r: r.llf); thz = zp.params.values
    piz = 1 / (1 + np.exp(-(Xi.values @ thz[:Xi.shape[1]]))); muz = np.exp(X.values @ thz[Xi.shape[1]:])
    R["zip"] = {"llf": float(zp.llf), "llf_partidas": [float(r.llf) for r in zips], "k": len(thz),
                "aic": float(-2 * zp.llf + 2 * len(thz)), "bic": float(-2 * zp.llf + np.log(n) * len(thz)),
                "conv": bool(zp.mle_retvals.get("converged")),
                "zeros_prev": float((piz + (1 - piz) * np.exp(-muz)).sum())}
    print("ZIP ok", R["zip"]["llf"], flush=True)
    # ------------------------------------------------------------ (5) ZINB: multiplos pontos de partida
    rng = np.random.default_rng(SEED)
    b_nb, g_lg = th_nb[:-1], lg.params.values
    partidas = [("padrao", None),
                ("NB", np.r_[[-3, 0, 0, 0, 0, 0], b_nb, alpha_nb]),
                ("logit+NB", np.r_[g_lg, b_nb, alpha_nb]),
                ("logit+Poisson+alpha0", np.r_[g_lg, glm.params.values, a0]),
                ("ZIP+alphaNB", np.r_[thz, alpha_nb])]
    for j in range(45):
        g = g_lg + rng.normal(0, 1.5, len(g_lg))
        b = b_nb.copy(); b[0] += rng.normal(0, 0.5); b[1:7] += rng.normal(0, 0.3, 6); b[7:] += rng.normal(0, 0.2, len(b) - 7)
        partidas.append((f"aleatoria_{j + 1:02d}", np.r_[g, b, float(np.exp(rng.uniform(0, np.log(30))))]))
    with Pool(2) as pool:
        ajustes = pool.map(ajusta_zinb, [sp for _, sp in partidas])
    lls = np.array([a["llf"] for a in ajustes])
    ib = int(np.argmax(lls)); th_z = np.array(ajustes[ib]["params"])
    zmod = ZeroInflatedNegativeBinomialP(y, X, exog_infl=Xi, p=2, inflation="logit")
    r_z = zmod.fit(start_params=th_z, method="bfgs", maxiter=5000, disp=0, skip_hessian=True)
    if r_z.llf > lls[ib]: th_z = r_z.params.values
    llz = float(zmod.loglike(th_z))
    H = hessiana_por_escore(zmod, th_z); ev = np.linalg.eigvalsh(-H)
    cov_z = np.linalg.inv(-H); S_z = zmod.score_obs(th_z)
    cp_z, Gp = cov_cluster(S_z, cov_z, par); cc_z, Gc = cov_cluster(S_z, cov_z, pais)
    ni = Xi.shape[1]; gz, bz, az = th_z[:ni], th_z[ni:-1], th_z[-1]
    pi = 1 / (1 + np.exp(-(Xi.values @ gz))); muzb = np.exp(X.values @ bz)
    p0 = pi + (1 - pi) * (1 + az * muzb) ** (-1 / az)
    d = zmod.loglikeobs(th_z) - nbm.loglikeobs(th_nb)
    kz = len(th_z)
    classe = lambda ll: ("otimo" if abs(ll - llz) < 1.0 else ("colapso_NB" if abs(ll - R["nb"]["llf"]) < 1.0 else "otimo_local"))
    R["zinb"] = {"llf": llz, "k": kz, "aic": -2 * llz + 2 * kz, "bic": -2 * llz + np.log(n) * kz, "alpha": float(az),
                 "mcfadden_vs_nb_nulo": float(1 - llz / nb_null.llf), "zeros_prev": float(p0.sum()),
                 "pi_media": float(pi.mean()), "pi_mediana": float(np.median(pi)),
                 "pi_media_hist1": float(pi[df["acordo_sanitario"] == 1].mean()),
                 "pi_media_hist0": float(pi[df["acordo_sanitario"] == 0].mean()),
                 "hess_eig_min": float(ev.min()), "hess_eig_max": float(ev.max()),
                 "grad_max": float(np.max(np.abs(zmod.score(th_z)))),
                 "vuong": float(np.sqrt(n) * d.mean() / d.std()),
                 "vuong_bic": float((d.sum() - (kz - len(th_nb)) * np.log(n) / 2) / (np.sqrt(n) * d.std())),
                 "G_par": int(Gp), "G_pais": int(Gc),
                 "partidas": [{"nome": nm, "llf": float(a["llf"]), "classe": classe(a["llf"])} for (nm, _), a in zip(partidas, ajustes)],
                 "tabela": tabela(list(zmod.exog_names), th_z, cov_z, cp_z, cc_z)}
    # elasticidade media do valor esperado E[Y] = (1 - pi) * mu em relacao a distancia
    jd, jdi = list(X.columns).index("ln_dist"), list(Xi.columns).index("ln_dist")
    R["zinb"]["elast_EY_dist"] = float(bz[jd] - (pi * gz[jdi]).mean())
    cont = pd.Series([p["classe"] for p in R["zinb"]["partidas"]]).value_counts().to_dict()
    R["zinb"]["contagem_classes"] = cont
    print("ZINB ok", llz, cont, flush=True)
    # ------------------------------------------------------------ (6) PPML
    gl = sm.GLM(y, X, family=sm.families.Poisson())
    rpp = gl.fit(maxiter=300, tol=1e-12, cov_type="cluster", cov_kwds={"groups": pd.factorize(par)[0]})
    rpc = gl.fit(maxiter=300, tol=1e-12, cov_type="cluster", cov_kwds={"groups": pd.factorize(pais)[0]})
    R["ppml"] = {"tabela": [{"var": v, "coef": float(rpp.params[v]), "se_par": float(rpp.bse[v]), "p_par": float(rpp.pvalues[v]),
                             "se_pais": float(rpc.bse[v]), "p_pais": float(rpc.pvalues[v])} for v in X.columns]}
    # ------------------------------------------------------------ (7) unidade de medida (milhoes)
    dfm, ym, Xm, Xim = carregar("painel_real_milhoes.csv")
    gm = sm.GLM(ym, Xm, family=sm.families.Poisson()).fit(maxiter=300, tol=1e-12)
    nbmm = NegativeBinomialP(ym, Xm, p=2)
    rnm = nbmm.fit(start_params=np.r_[gm.params.values, alpha_momentos(ym.values)], method="bfgs", maxiter=5000, disp=0)
    rnm = nbmm.fit(start_params=rnm.params.values, method="newton", maxiter=100, disp=0)
    zmm = ZeroInflatedNegativeBinomialP(ym, Xm, exog_infl=Xim, p=2, inflation="logit")
    lgm = sm.Logit((ym == 0).astype(float), Xim).fit(disp=0, method="bfgs", maxiter=2000)
    th_start_m = th_z.copy(); th_start_m[ni] -= np.log(1000)
    cand = [np.r_[lgm.params.values, rnm.params.values], th_start_m]
    rngm = np.random.default_rng(SEED + 1)
    for j in range(10):
        g = lgm.params.values + rngm.normal(0, 1.5, ni)
        b = rnm.params.values[:-1].copy(); b[1:7] += rngm.normal(0, 0.3, 6)
        cand.append(np.r_[g, b, float(np.exp(rngm.uniform(0, np.log(30))))])
    fits_m = []
    for sp in cand:
        try: fits_m.append(zmm.fit(start_params=sp, method="bfgs", maxiter=6000, disp=0, skip_hessian=True))
        except Exception: pass
    zbm = max(fits_m, key=lambda r: r.llf)
    arq_ct = "comtrade_exportacoes_br.csv" if os.path.exists("comtrade_exportacoes_br.csv") else os.path.join("dados_reais", "comtrade", "comtrade_exportacoes_br.csv")
    raw = pd.read_csv(arq_ct, dtype={"cmdCode": str})
    raw = raw[raw["partnerISO"] != "W00"].copy(); raw["cmdCode"] = raw["cmdCode"].str.zfill(6)
    HS = {"120190": "soja_grao", "230400": "farelo_soja", "150710": "oleo_soja", "020230": "carne_bovina",
          "020714": "carne_frango", "090111": "cafe", "170114": "acucar", "200911": "suco_laranja",
          "100590": "milho", "520100": "algodao"}
    raw["commodity"] = raw["cmdCode"].map(HS)
    gr = raw.groupby(["partnerISO", "commodity", "period"], as_index=False)["primaryValue"].sum().rename(
        columns={"partnerISO": "pais", "period": "ano"})
    mm = df[["pais", "commodity", "ano"]].merge(gr, on=["pais", "commodity", "ano"], how="left").fillna({"primaryValue": 0})
    bruto = mm["primaryValue"].values; posb = bruto > 0
    R["unidade"] = {"fluxos_brutos_pos": int(posb.sum()),
                    "destruidos_mil": int((posb & (np.round(bruto / 1e3) == 0)).sum()),
                    "destruidos_milhao": int((posb & (np.round(bruto / 1e6) == 0)).sum()),
                    "valor_destruido_mil_usd": float(bruto[posb & (np.round(bruto / 1e3) == 0)].sum()),
                    "valor_destruido_milhao_usd": float(bruto[posb & (np.round(bruto / 1e6) == 0)].sum()),
                    "valor_total_usd": float(bruto.sum()), "pct_zeros_milhao": float((ym == 0).mean()),
                    "ppml_mil": {v: float(glm.params[v]) for v in ["const"] + GRAV},
                    "ppml_milhao": {v: float(gm.params[v]) for v in ["const"] + GRAV},
                    "nb_milhao": {v: float(rnm.params[v]) for v in ["const"] + GRAV + ["alpha"]},
                    "zinb_milhao": dict(zip(zbm.params.index, map(float, zbm.params.values))),
                    "zinb_milhao_llf": float(zbm.llf), "zinb_milhao_llfs": [float(r.llf) for r in fits_m]}
    print("Unidade ok", flush=True)
    # ------------------------------------------------------------ (8) K-Means + multinivel
    feat = df.groupby("pais").agg(ln_pib=("ln_pib_parc", "mean"), ln_pibpc=("ln_pib_pc_parc", "mean"),
                                  ln_dist=("ln_dist", "mean")).reset_index()
    Z = StandardScaler().fit_transform(feat[["ln_pib", "ln_pibpc", "ln_dist"]])
    km_res = [KMeans(n_clusters=k, n_init=50, random_state=42).fit(Z) for k in range(2, 9)]
    R["kmeans"] = {"k": list(range(2, 9)), "inercia": [float(m.inertia_) for m in km_res],
                   "silhueta": [float(silhouette_score(Z, m.labels_)) for m in km_res]}
    feat["tipo"] = km_res[2].labels_                                                # K = 4
    meta = json.load(open("painel_real_meta_milhares.json"))["centros_preditores"]
    perf = feat.groupby("tipo").agg(n=("pais", "size"), ln_pib=("ln_pib", "mean"), ln_pibpc=("ln_pibpc", "mean"), ln_dist=("ln_dist", "mean"))
    perf["pib_bi"] = np.exp(perf["ln_pib"] + meta["ln_pib_parc"]) / 1e9
    perf["pibpc"] = np.exp(perf["ln_pibpc"] + meta["ln_pib_pc_parc"])
    perf["dist_km"] = np.exp(perf["ln_dist"] + meta["ln_dist"])
    d2 = df.merge(feat[["pais", "tipo"]], on="pais")
    d2["pi_zinb"] = pi
    perf["pi_medio"] = d2.groupby("tipo")["pi_zinb"].mean()
    perf["pct_zeros"] = d2.groupby("tipo")["valor_exportado"].apply(lambda s: float((s == 0).mean()))
    R["kmeans"]["perfil"] = perf.reset_index().to_dict("records")
    R["kmeans"]["membros"] = feat[["pais", "tipo"]].to_dict("records")
    FULL = ["ln_pib_parc", "ln_pib_pc_parc", "ln_dist", "rta", "idioma_comum", "acordo_sanitario", "cambio", "preco_lag"]
    dc2 = pd.get_dummies(d2["commodity"], prefix="comm", drop_first=True, dtype=float)
    da2 = pd.get_dummies(d2["ano"], prefix="ano", drop_first=True, dtype=float)
    dt2 = pd.get_dummies(d2["tipo"], prefix="tipo", drop_first=True, dtype=float)
    ybin = (d2["valor_exportado"] > 0).astype(float).values
    esp = {"nulo": sm.add_constant(pd.DataFrame(index=d2.index)), "tipologia": sm.add_constant(dt2),
           "completo": sm.add_constant(pd.concat([d2[FULL].astype(float), dc2, da2], axis=1))}
    R["glmm"] = {}; th_nulo = None
    for nm, Xs in esp.items():
        mod = GLMMLogitAGQ(ybin, Xs.values, d2["pais"].values, nq=15)
        st = np.r_[0.0, 0.5] if nm == "nulo" else np.r_[np.asarray(sm.Logit(ybin, Xs.values).fit(disp=0, method="bfgs", maxiter=3000).params), th_nulo[-1]]
        mod.fit(start=st)
        if nm == "nulo": th_nulo = mod.theta.copy()
        s2u = float(np.exp(2 * mod.theta[-1]))
        R["glmm"][nm] = {"llf": float(mod.llf), "sigma2_u": s2u, "icc": s2u / (s2u + np.pi ** 2 / 3),
                         "coef": {c: {"b": float(mod.theta[i]), "se": float(mod.se[i]),
                                      "p": float(2 * stats.norm.sf(abs(mod.theta[i] / mod.se[i])))} for i, c in enumerate(Xs.columns)}}
    pos2 = d2[d2["valor_exportado"] > 0].copy(); pos2["ln_y"] = np.log(pos2["valor_exportado"])
    dcp = pd.get_dummies(pos2["commodity"], prefix="comm", drop_first=True, dtype=float)
    dap = pd.get_dummies(pos2["ano"], prefix="ano", drop_first=True, dtype=float)
    dtp = pd.get_dummies(pos2["tipo"], prefix="tipo", drop_first=True, dtype=float)
    espH = {"nulo": sm.add_constant(pd.DataFrame(index=pos2.index)), "tipologia": sm.add_constant(dtp),
            "completo": sm.add_constant(pd.concat([pos2[FULL].astype(float), dcp, dap], axis=1))}
    R["hlm"] = {}
    for nm, Xs in espH.items():
        r = MixedLM(pos2["ln_y"], Xs, groups=pos2["pais"]).fit(reml=False, method="bfgs", maxiter=5000)
        t2 = float(np.asarray(r.cov_re)[0, 0]); s2 = float(r.scale)
        R["hlm"][nm] = {"llf": float(r.llf), "tau2": t2, "sigma2": s2, "icc": t2 / (t2 + s2), "n": int(len(pos2)),
                        "coef": {c: {"b": float(r.fe_params[c]), "se": float(r.bse_fe[c]), "p": float(r.pvalues[c])} for c in Xs.columns}}
    R["tempo_total_s"] = time.time() - T0
    json.dump(R, open("resultados_finais.json", "w"), indent=1, ensure_ascii=False, default=float)
    print("Concluido em", round(R["tempo_total_s"]), "s", flush=True)
