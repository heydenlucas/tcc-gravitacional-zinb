"""
montar_paineis_alt.py
=====================
Gera as versoes do painel que diferem apenas na unidade da variavel dependente:
  - painel_real_milhares.csv  -> Y = round(primaryValue / 1e3)  (milhares de US$; painel principal do TCC)
  - painel_real_toneladas.csv -> Y = round(netWgt / 1e3)        (toneladas; nao utilizado no TCC)
O painel em milhoes de US$ (painel_real_milhoes.csv) e gerado por montar_painel_real.py.
Os preditores sao os mesmos em todas as versoes; reutiliza as funcoes de montar_painel_real.py.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import montar_painel_real as mpr

BASE = Path(__file__).resolve().parent
DR = next((p / "dados_reais" for p in (BASE, BASE.parent, BASE.parent.parent) if (p / "dados_reais").exists()),
          BASE / "dados_reais")   # pasta com os dados brutos (UN Comtrade, CEPII, WDI, Pink Sheet)
OUT = BASE   # paineis gravados na propria pasta da analise

ANOS = mpr.ANOS
HS6_MAP = mpr.HS6_MAP
CONTINUAS = mpr.CONTINUAS


def y_comtrade_alt(metrica: str, divisor: float) -> pd.DataFrame:
    """Le o Comtrade e devolve Y na unidade pedida.
    metrica: 'primaryValue' (monetario) ou 'netWgt' (peso liquido)
    divisor: para milhares=1e3, milhoes=1e6, toneladas (kg)=1e3
    """
    c = pd.read_csv(DR / "comtrade" / "comtrade_exportacoes_br.csv")
    c = c[c["partnerISO"] != "W00"].copy()
    c["cmdCode"] = c["cmdCode"].astype(str).str.zfill(6)
    c = c[c["cmdCode"].isin(HS6_MAP)]
    c["commodity"] = c["cmdCode"].map(lambda x: HS6_MAP[x][0])
    c[metrica] = c[metrica].fillna(0.0)
    g = c.groupby(["partnerISO", "commodity", "period"], as_index=False)[metrica].sum()
    g = g.rename(columns={"partnerISO": "pais", "period": "ano"})
    g["valor_exportado"] = (g[metrica] / divisor).round().astype("int64")
    return g[["pais", "commodity", "ano", "valor_exportado"]]


def montar_painel(metrica: str, divisor: float, sufixo: str, unidade_desc: str):
    """Replica a logica de montar_painel_real.main() com Y na unidade indicada."""
    print(f"\n>> Gerando painel {sufixo} ({unidade_desc})")
    y = y_comtrade_alt(metrica, divisor)
    san = mpr.proxy_sanitario()
    grav = mpr.gravity_cepii()
    pib = mpr.wdi_pib()
    camb = mpr.wdi_cambio()
    preco = mpr.preco_lag_pink()

    parceiros = sorted(grav["pais"].unique())
    commodities = [v[0] for v in HS6_MAP.values()]
    grade = pd.MultiIndex.from_product(
        [parceiros, commodities, ANOS],
        names=["pais", "commodity", "ano"]
    ).to_frame(index=False)

    df = (grade
          .merge(y, on=["pais", "commodity", "ano"], how="left")
          .merge(grav, on=["pais", "ano"], how="left")
          .merge(pib, on=["pais", "ano"], how="left")
          .merge(camb, on=["pais", "ano"], how="left")
          .merge(preco, on=["commodity", "ano"], how="left")
          .merge(san, on=["pais", "commodity"], how="left"))

    df["valor_exportado"] = df["valor_exportado"].fillna(0).astype("int64")
    df["acordo_sanitario"] = df["acordo_sanitario"].fillna(0).astype(int)
    df["cambio"] = df["cambio"].fillna(0.0)
    df["idioma_comum"] = df["idioma_comum"].fillna(0).astype(int)
    df["rta"] = df["rta"].fillna(0).astype(int)

    df = df.dropna(subset=["ln_pib_parc", "ln_pib_pc_parc", "ln_dist", "preco_lag"])

    centros = {c: float(df[c].mean()) for c in CONTINUAS}
    for c in CONTINUAS:
        df[c] = df[c] - centros[c]

    cols = ["pais", "commodity", "ano", "ln_pib_parc", "ln_pib_pc_parc", "ln_dist",
            "idioma_comum", "rta", "acordo_sanitario", "cambio", "preco_lag",
            "valor_exportado"]
    df = df[cols].drop_duplicates(["pais", "commodity", "ano"]).reset_index(drop=True)

    out_csv = OUT / f"painel_real_{sufixo}.csv"
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
        "Y_unidade": unidade_desc,
        "pct_zeros": float((yv == 0).mean()),
        "media": float(yv.mean()),
        "variancia": float(yv.var()),
        "razao_var_media": float(yv.var() / yv.mean()) if yv.mean() else float("nan"),
        "maximo": int(yv.max()),
        "nota": (f"Painel com Y em {unidade_desc}. "
                 "Preditores e estrutura identicos aos das demais versoes do painel."),
    }
    with open(OUT / f"painel_real_meta_{sufixo}.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    print(f"   N = {len(df)} | zeros = {(yv == 0).mean():.1%} | "
          f"media = {yv.mean():.2f} | var = {yv.var():.1f} | "
          f"var/media = {meta['razao_var_media']:.1f} | max = {yv.max()}")
    print(f"   salvo: {out_csv.name}")
    return df, meta


if __name__ == "__main__":
    montar_painel("primaryValue", 1e3, "milhares",
                  "valor exportado em milhares de US$ (arredondado)")
    montar_painel("netWgt",       1e3, "toneladas",
                  "peso liquido exportado em toneladas (arredondado)")
    print("\n>> Concluido.")
