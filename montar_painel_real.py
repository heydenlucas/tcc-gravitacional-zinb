"""
montar_painel_real.py
=====================
Integra as quatro fontes publicas de dados_reais/ no painel parceiro-commodity-ano
do TCC "Modelo gravitacional com inflacao de zeros para as exportacoes
agropecuarias brasileiras (2014-2024)" (MBA em Data Science e Analytics, USP/Esalq).

Fontes (em dados_reais/):
  - UN Comtrade           -> valor exportado e historico de exportacao (2000-2013)
  - CEPII Gravity         -> distancia, idioma comum e acordo comercial regional
  - World Bank WDI        -> PIB, PIB per capita e taxas de cambio
  - World Bank Pink Sheet -> preco internacional da commodity (defasado em um ano)

Saida: painel_real_milhoes.csv e painel_real_meta_milhoes.json, com o valor
exportado em milhoes de US$ (escala usada na analise de sensibilidade). O painel
principal, em milhares de US$, e gerado por montar_paineis_alt.py, que reutiliza
as funcoes deste modulo.

Decisoes de construcao:
  - acordo_sanitario = 1 se o Brasil exportou o produto ao parceiro em 2000-2013
        (historico de exportacao, aproximacao do acesso previo ao mercado).
  - cambio = variacao acumulada, desde 2014, do logaritmo do cambio bilateral do
        real em relacao a moeda do parceiro (triangulacao pelo dolar); valores
        positivos indicam depreciacao relativa do real.
  - CEPII vai ate 2021: para 2022-2024 repete-se o valor de 2021 de cada par.
  - Retem-se apenas registros de entidades vigentes na base do CEPII.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DR = next((p / "dados_reais" for p in (BASE, BASE.parent, BASE.parent.parent) if (p / "dados_reais").exists()),
          BASE / "dados_reais")   # pasta com os dados brutos (UN Comtrade, CEPII, WDI, Pink Sheet)
OUT = BASE   # paineis gravados na propria pasta da analise
OUT.mkdir(exist_ok=True)

ANOS = list(range(2014, 2025))   # 2014..2024 -- janela do modelo

# HS-6 -> (nome da commodity, nome da coluna no Pink Sheet 'Annual Prices (Nominal)')
# obs: o cmdCode da Comtrade perde zeros a esquerda -> padronizamos com zfill(6).
HS6_MAP = {
    "120190": ("soja_grao",    "Soybeans"),
    "230400": ("farelo_soja",  "Soybean meal"),
    "150710": ("oleo_soja",    "Soybean oil"),
    "020230": ("carne_bovina", "Beef"),
    "020714": ("carne_frango", "Chicken"),
    "090111": ("cafe",         "Coffee, Arabica"),
    "170114": ("acucar",       "Sugar, world"),
    "200911": ("suco_laranja", "Orange"),
    "100590": ("milho",        "Maize"),
    "520100": ("algodao",      "Cotton, A Index"),
}
CONTINUAS = ["ln_pib_parc", "ln_pib_pc_parc", "ln_dist", "cambio", "preco_lag"]


# ---------------------------------------------------------------------------
def y_comtrade() -> pd.DataFrame:
    """Y = valor exportado em milhoes de US$ (escala de sensibilidade) por pais x commodity x ano."""
    c = pd.read_csv(DR / "comtrade" / "comtrade_exportacoes_br.csv")
    c = c[c["partnerISO"] != "W00"].copy()                  # exclui agregado Mundo
    c["cmdCode"] = c["cmdCode"].astype(str).str.zfill(6)
    c = c[c["cmdCode"].isin(HS6_MAP)]
    c["commodity"] = c["cmdCode"].map(lambda x: HS6_MAP[x][0])
    g = c.groupby(["partnerISO", "commodity", "period"], as_index=False)["primaryValue"].sum()
    g = g.rename(columns={"partnerISO": "pais", "period": "ano"})
    g["valor_exportado"] = (g["primaryValue"] / 1e6).round().astype("int64")
    return g[["pais", "commodity", "ano", "valor_exportado"]]


def proxy_sanitario() -> pd.DataFrame:
    """acordo_sanitario = 1 se o Brasil exportou o par (pais,commodity) em 2000-2013."""
    h = pd.read_csv(DR / "comtrade" / "comtrade_historico_2000_2024.csv")
    h = h[(h["partnerISO"] != "W00") & (h["period"] <= 2013)].copy()
    h["cmdCode"] = h["cmdCode"].astype(str).str.zfill(6)
    h = h[h["cmdCode"].isin(HS6_MAP) & (h["primaryValue"] > 0)]
    h["commodity"] = h["cmdCode"].map(lambda x: HS6_MAP[x][0])
    pares = h[["partnerISO", "commodity"]].drop_duplicates()
    pares = pares.rename(columns={"partnerISO": "pais"})
    pares["acordo_sanitario"] = 1
    return pares


def gravity_cepii() -> pd.DataFrame:
    """distancia, idioma comum e RTA por pais x ano (com carry-forward 2022-2024)."""
    g = pd.read_csv(DR / "cepii" / "gravity_brasil.csv")
    g = g[(g["year"] >= 2014) & (g["iso3_d"] != "BRA")].copy()
    # CORRECAO (set/2026): o CEPII traz, para alguns ISO3, uma linha da entidade
    # historica (country_id ".1", country_exists_d = 0, sem distancia) antes da
    # linha valida. O drop_duplicates abaixo mantinha a primeira (vazia) e o pais
    # era descartado por falta de distancia (ex.: DEU, IDN, VNM, MYS, PAK).
    g = g[g["country_exists_d"] == 1].copy()
    g = g.rename(columns={"iso3_d": "pais", "year": "ano"})
    g["ln_dist"] = np.log(g["dist"])
    g["idioma_comum"] = g["comlang_off"].fillna(0).astype(int)
    g["rta"] = g["fta_wto"].fillna(0).astype(int)
    g = g[["pais", "ano", "ln_dist", "idioma_comum", "rta"]].drop_duplicates(["pais", "ano"])
    ult = g[g["ano"] == 2021].copy()
    for a in (2022, 2023, 2024):
        tmp = ult.copy(); tmp["ano"] = a
        g = pd.concat([g, tmp], ignore_index=True)
    return g


def _wdi_long(arquivo: str, nome: str) -> pd.DataFrame:
    """Le um indicador WDI (formato largo) e devolve formato longo pais x ano."""
    w = pd.read_csv(DR / "world_bank_wdi" / arquivo, skiprows=4)
    anos_str = [str(a) for a in ANOS]
    w = w[["Country Code"] + anos_str]
    longo = w.melt(id_vars="Country Code", var_name="ano", value_name=nome)
    longo["ano"] = longo["ano"].astype(int)
    return longo.rename(columns={"Country Code": "pais"})


def wdi_pib() -> pd.DataFrame:
    """ln(PIB) e ln(PIB per capita) do parceiro."""
    pib = _wdi_long("NY.GDP.MKTP.CD.csv", "pib")
    pibpc = _wdi_long("NY.GDP.PCAP.CD.csv", "pibpc")
    m = pib.merge(pibpc, on=["pais", "ano"], how="outer")
    m["ln_pib_parc"] = np.log(m["pib"])
    m["ln_pib_pc_parc"] = np.log(m["pibpc"])
    return m[["pais", "ano", "ln_pib_parc", "ln_pib_pc_parc"]]


def wdi_cambio() -> pd.DataFrame:
    """cambio = competitividade cambial bilateral do BRL vs o parceiro."""
    longo = _wdi_long("PA.NUS.FCRF.csv", "fcrf")
    base = longo[longo["ano"] == 2014][["pais", "fcrf"]].rename(columns={"fcrf": "fcrf0"})
    longo = longo.merge(base, on="pais", how="left")
    longo["dlog"] = np.log(longo["fcrf"]) - np.log(longo["fcrf0"])
    bra = longo[longo["pais"] == "BRA"][["ano", "dlog"]].rename(columns={"dlog": "dlog_bra"})
    longo = longo.merge(bra, on="ano", how="left")
    longo["cambio"] = longo["dlog_bra"] - longo["dlog"]
    return longo[["pais", "ano", "cambio"]]


def preco_lag_pink() -> pd.DataFrame:
    """preco_lag = ln(preco da commodity no ano anterior), do Pink Sheet."""
    ap = pd.read_excel(DR / "world_bank_pink_sheet" / "wb_pink_sheet_annual.xlsx",
                       sheet_name="Annual Prices (Nominal)", header=None)
    nomes = ap.iloc[6].tolist()
    col_de = {n: i for i, n in enumerate(nomes) if pd.notna(n)}
    dados = ap.iloc[8:].copy()
    dados[0] = pd.to_numeric(dados[0], errors="coerce")
    regs = []
    for hs, (comm, ps_nome) in HS6_MAP.items():
        ci = col_de[ps_nome]
        serie = dados[[0, ci]].copy()
        serie.columns = ["ano", "preco"]
        serie["preco"] = pd.to_numeric(serie["preco"], errors="coerce")
        serie = serie.dropna()
        for _, r in serie.iterrows():
            regs.append({"commodity": comm, "ano": int(r["ano"]) + 1,
                         "preco_lag": float(np.log(r["preco"]))})
    return pd.DataFrame(regs)


# ---------------------------------------------------------------------------
def main():
    print(">> Carregando as 4 fontes...")
    y = y_comtrade()
    san = proxy_sanitario()
    grav = gravity_cepii()
    pib = wdi_pib()
    camb = wdi_cambio()
    preco = preco_lag_pink()
    print(f"   Comtrade Y: {len(y)} fluxos positivos")
    print(f"   proxy sanitario: {len(san)} pares (pais,commodity) com acesso historico")
    print(f"   gravity: {grav['pais'].nunique()} parceiros")

    # --- grade completa: parceiros (do gravity) x commodities x anos ---
    parceiros = sorted(grav["pais"].unique())
    commodities = [v[0] for v in HS6_MAP.values()]
    grade = pd.MultiIndex.from_product([parceiros, commodities, ANOS],
                                       names=["pais", "commodity", "ano"]).to_frame(index=False)
    print(f"   grade inicial: {len(grade)} celulas ({len(parceiros)} paises x "
          f"{len(commodities)} commodities x {len(ANOS)} anos)")

    # --- junta tudo ---
    df = (grade
          .merge(y, on=["pais", "commodity", "ano"], how="left")
          .merge(grav, on=["pais", "ano"], how="left")
          .merge(pib, on=["pais", "ano"], how="left")
          .merge(camb, on=["pais", "ano"], how="left")
          .merge(preco, on=["commodity", "ano"], how="left")
          .merge(san, on=["pais", "commodity"], how="left"))

    # --- preenchimentos ---
    df["valor_exportado"] = df["valor_exportado"].fillna(0).astype("int64")  # celula sem fluxo = 0
    df["acordo_sanitario"] = df["acordo_sanitario"].fillna(0).astype(int)
    df["cambio"] = df["cambio"].fillna(0.0)          # parceiro sem serie de cambio = sem variacao
    df["idioma_comum"] = df["idioma_comum"].fillna(0).astype(int)
    df["rta"] = df["rta"].fillna(0).astype(int)

    # --- descarta linhas sem os preditores criticos ---
    antes = len(df)
    df = df.dropna(subset=["ln_pib_parc", "ln_pib_pc_parc", "ln_dist", "preco_lag"])
    print(f"   apos descartar linhas sem PIB/distancia/preco: {len(df)} (de {antes})")

    # --- centragem dos preditores continuos (estabiliza o otimizador) ---
    centros = {c: float(df[c].mean()) for c in CONTINUAS}
    for c in CONTINUAS:
        df[c] = df[c] - centros[c]

    # --- ordem final de colunas ---
    cols = ["pais", "commodity", "ano", "ln_pib_parc", "ln_pib_pc_parc", "ln_dist",
            "idioma_comum", "rta", "acordo_sanitario", "cambio", "preco_lag",
            "valor_exportado"]
    df = df[cols].drop_duplicates(["pais", "commodity", "ano"]).reset_index(drop=True)

    out_csv = OUT / "painel_real_milhoes.csv"   # escala de controle (milhoes de US$)
    df.to_csv(out_csv, index=False)

    yv = df["valor_exportado"].to_numpy()
    meta = {
        "is_synthetic": False,
        "fonte": "UN Comtrade + CEPII Gravity + World Bank WDI + World Bank Pink Sheet",
        "n_obs": int(len(df)),
        "n_paises": int(df["pais"].nunique()),
        "n_commodities": int(df["commodity"].nunique()),
        "anos": ANOS,
        "centros_preditores": centros,
        "Y_unidade": "valor exportado em milhoes de US$ (arredondado)",
        "pct_zeros": float((yv == 0).mean()),
        "nota": "acordo_sanitario = proxy 'exportou em 2000-2013'; cambio = "
                "competitividade bilateral do BRL; CEPII 2022-2024 = carry-forward de 2021.",
    }
    with open(OUT / "painel_real_meta_milhoes.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    print(f"\n>> Painel real salvo: {out_csv}")
    print(f"   {len(df)} obs | {df['pais'].nunique()} paises | "
          f"{df['commodity'].nunique()} commodities | {df['ano'].min()}-{df['ano'].max()}")
    print(f"   zeros: {(yv == 0).mean():.1%} | media(Y): {yv.mean():.2f} | "
          f"var(Y): {yv.var():.1f} | razao var/media: {yv.var()/yv.mean():.1f} | max: {yv.max()}")
    print(f"   linhas com fluxo > 0: {(yv > 0).sum()}")


if __name__ == "__main__":
    main()
