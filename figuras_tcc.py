from figstyle import *
import pandas as pd, numpy as np, json
setup()
R = json.load(open("resultados_finais.json"))
df = pd.read_csv("painel_real_milhares.csv"); y = df["valor_exportado"].values
NOMES = {"soja_grao":"Soja em grão","farelo_soja":"Farelo de soja","oleo_soja":"Óleo de soja","carne_bovina":"Carne bovina",
         "carne_frango":"Carne de frango","cafe":"Café","acucar":"Açúcar","suco_laranja":"Suco de laranja","milho":"Milho","algodao":"Algodão"}
# ---------------- Figura 1: distribuicao ----------------
fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 6.2*CM), gridspec_kw={"width_ratios":[1.0, 1.25, 1.25]})
ax = axes[0]; nz, npos = int((y==0).sum()), int((y>0).sum())
bars = ax.bar([0, 1], [nz, npos], color=["0.25", "0.65"], width=0.6)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Zero", "Positivo"]); ax.set_ylabel("Número de observações")
for b, v in zip(bars, [nz, npos]):
    ax.text(b.get_x()+b.get_width()/2, v+300, f"{fmt_br(v)}\n({fmt_br(100*v/len(y),1)}%)", ha="center", va="bottom", fontsize=7.5)
ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 19500); br_axis(ax, "y"); panel(ax, "A")
ax = axes[1]; lv = np.log10(y[y>0])
ax.hist(lv, bins=40, color="0.55", edgecolor="white", linewidth=0.3)
med = np.median(y[y>0]); ax.axvline(np.log10(med), color="black", linestyle="--", linewidth=1.0)
ax.set_ylim(0, 470)
ax.text(np.log10(med)+0.15, 455, f"Mediana:\nUS\\$ {fmt_br(med/1000,1)} milhões", fontsize=7.5, va="top", ha="left")
ax.set_xlabel("log$_{10}$ do valor exportado (US\\$ mil)"); ax.set_ylabel("Frequência")
br_axis(ax, "y"); br_axis(ax, "x"); panel(ax, "B")
ax = axes[2]
pz = df.assign(z=(df.valor_exportado==0)).groupby("commodity")["z"].mean().sort_values()
ax.barh([NOMES[c] for c in pz.index], 100*pz.values, color="0.45", height=0.65)
for i, v in enumerate(100*pz.values): ax.text(v+1.5, i, f"{fmt_br(v,0)}%", va="center", fontsize=7)
ax.set_xlabel("Fluxos nulos (%)"); ax.set_xlim(0, 108); ax.tick_params(axis="y", length=0)
panel(ax, "C", x=-0.02)
fig.tight_layout(w_pad=0.6); save(fig, "fig1_distribuicao")
# ---------------- Figura 2: media x variancia ----------------
alpha = R["nb"]["alpha"]
cel = df.groupby(["commodity","ano"])["valor_exportado"].agg(["mean", "var"])
fig, ax = plt.subplots(figsize=(11.5*CM, 7.6*CM))
ax.scatter(cel["mean"], cel["var"], s=10, color="0.35", edgecolor="none", label="Células commodity-ano")
xx = np.logspace(0, np.log10(cel["mean"].max()*1.3), 200)
ax.plot(xx, xx, color="black", linestyle="--", linewidth=1.0, label="Poisson: Var = μ")
ax.plot(xx, xx + alpha*xx**2, color="black", linewidth=1.0, label=f"Binomial Negativa: Var = μ + αμ², α = {fmt_br(alpha,2)}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Média do valor exportado por célula (US\\$ mil)"); ax.set_ylabel("Variância por célula (US\\$ mil)²")
ax.legend(loc="upper left", handlelength=2.2, fontsize=8); fig.tight_layout(); save(fig, "fig2_media_variancia")
# ---------------- Figura 3: multiplas partidas + zeros ----------------
part = pd.DataFrame(R["zinb"]["partidas"]).sort_values("llf", ascending=False).reset_index(drop=True)
fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 5.7*CM), gridspec_kw={"width_ratios":[1.35, 1]})
ax = axes[0]
mk = {"otimo": ("o", "black", "Máximo de maior verossimilhança"), "otimo_local": ("s", "0.55", "Máximo local inferior"),
      "colapso_NB": ("^", "white", "Solução degenerada (π → 0)")}
for cl, (m, c, lab) in mk.items():
    s = part[part["classe"] == cl]
    ax.scatter(s.index+1, s["llf"]/1000, marker=m, s=18, facecolor=c, edgecolor="black", linewidth=0.7, label=f"{lab} ({len(s)})", zorder=3)
ax.axhline(R["nb"]["llf"]/1000, color="black", linestyle=":", linewidth=1.0)
ax.text(1, R["nb"]["llf"]/1000+0.12, "Binomial Negativa", fontsize=7.5, va="bottom")
ax.set_xlabel("Ponto de partida (ordenado pela log-verossimilhança)"); ax.set_ylabel("Log-verossimilhança (mil)")
ax.set_ylim(-93.2, -87.0); br_axis(ax, "y", 1)
ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.42), handletextpad=0.3, fontsize=7.5)
panel(ax, "A")
ax = axes[1]
obs = R["eda"]["n_zeros"]
mods = ["Poisson", "NB", "ZIP", "ZINB"]; vals = [R["poisson"]["zeros_prev"], R["nb"]["zeros_prev"], R["zip"]["zeros_prev"], R["zinb"]["zeros_prev"]]
b = ax.bar(range(4), vals, color=["0.8", "0.6", "0.45", "0.25"], width=0.6, edgecolor="black", linewidth=0.5)
ax.axhline(obs, color="black", linestyle="--", linewidth=1.0)
ax.text(-0.35, obs+300, f"Observados: {fmt_br(obs)}", fontsize=7.5, va="bottom")
for i, v in enumerate(vals): ax.text(i, max(v, 300)+150 if i==0 else v-2400, fmt_br(v), ha="center", va="bottom", fontsize=7.5, color="black" if i<2 else "white")
ax.set_xticks(range(4)); ax.set_xticklabels(mods); ax.set_ylabel("Zeros previstos"); ax.set_ylim(0, 17500)
br_axis(ax, "y"); panel(ax, "B")
fig.tight_layout(w_pad=1.5); save(fig, "fig3_partidas_zeros")
# ---------------- Figura 4: coeficientes do ZINB ----------------
tab = {t["var"]: t for t in R["zinb"]["tabela"]}
LAB = {"ln_pib_parc":"PIB do parceiro (ln)","ln_dist":"Distância (ln)","rta":"Acordo comercial regional","idioma_comum":"Idioma comum",
       "cambio":"Câmbio bilateral","preco_lag":"Preço defasado (ln)","ln_pib_pc_parc":"PIB per capita (ln)","acordo_sanitario":"Histórico de exportação"}
cont = ["ln_pib_parc","ln_dist","rta","idioma_comum","cambio","preco_lag"]
infl = ["ln_pib_pc_parc","acordo_sanitario","ln_dist","idioma_comum","rta"]
fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 6.6*CM))
for ax, vars_, pre, letter, xlab in [(axes[0], cont, "", "A", "Coeficiente estimado"),
                                      (axes[1], infl, "inflate_", "B", "Coeficiente estimado")]:
    for i, v in enumerate(vars_[::-1]):
        t = tab[pre + v]; lo, hi = t["coef"]-1.96*t["se_par"], t["coef"]+1.96*t["se_par"]
        ax.plot([lo, hi], [i, i], color="black", linewidth=1.2)
        ax.plot(t["coef"], i, "o", color="black", markersize=4)
    ax.axvline(0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(vars_))); ax.set_yticklabels([LAB[v] for v in vars_[::-1]]); ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlab); panel(ax, letter, x=-0.02)
    br_axis(ax, "x", 0 if pre else 1)
    if pre: ax.set_xlim(-3.9, 1.0); ax.set_xticks([-3, -2, -1, 0, 1])
    else: ax.set_xlim(-1.1, 1.3); ax.set_xticks([-1, -0.5, 0, 0.5, 1])
fig.tight_layout(w_pad=2.0); save(fig, "fig4_coef_zinb")
# ---------------- Figura 5: K-Means ----------------
km = R["kmeans"]
fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 5.6*CM))
ax = axes[0]; ax.plot(km["k"], km["inercia"], "o-", color="black", linewidth=1.2, markersize=4)
ax.plot(4, km["inercia"][2], "o", markersize=8, markerfacecolor="none", markeredgecolor="black")
ax.set_xlabel("Número de grupos (K)"); ax.set_ylabel("Inércia intragrupos"); br_axis(ax, "y"); panel(ax, "A")
ax = axes[1]; ax.plot(km["k"], km["silhueta"], "o-", color="black", linewidth=1.2, markersize=4)
ax.plot(4, km["silhueta"][2], "o", markersize=8, markerfacecolor="none", markeredgecolor="black")
ax.set_xlabel("Número de grupos (K)"); ax.set_ylabel("Silhueta média"); br_axis(ax, "y", 2); panel(ax, "B")
fig.tight_layout(w_pad=2.0); save(fig, "fig5_kmeans")
print("figuras ok")
