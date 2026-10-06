"""
baixar_comtrade.py
==================
Baixa as exportacoes agropecuarias do Brasil da UN Comtrade e salva em
dados_reais/comtrade/comtrade_exportacoes_br.csv, no formato usado por montar_painel_real.py.

COMO USAR (rode na SUA maquina, onde a internet funciona):
  1. pip install comtradeapicall pandas
  2. Informe sua chave gratuita da API em dados_reais/comtrade/chave_api.txt
     (linha PRIMARY=...) ou na variavel de ambiente COMTRADE_KEY. A chave nao
     e distribuida neste repositorio (ver .gitignore).
  3. python baixar_comtrade.py
     -> salva dados_reais/comtrade/comtrade_exportacoes_br.csv (janela principal)
     -> salva dados_reais/comtrade/comtrade_historico_2000_2024.csv (para o proxy
        de barreira sanitaria 'nunca-exportou'); desative em BAIXAR_HISTORICO.

Conta gratuita: 500 chamadas/dia, 100 mil registros por chamada -- folga de sobra
para este recorte (1 chamada por ano).
"""

import os
import time
from pathlib import Path

import pandas as pd

try:
    import comtradeapicall
except ImportError:
    raise SystemExit("Instale o pacote: pip install comtradeapicall pandas")

# ---------------------------------------------------------------------------
# CONFIGURACAO
# ---------------------------------------------------------------------------
def _carregar_chave() -> str:
    """
    Le a chave de API do arquivo dados_reais/comtrade/chave_api.txt (linha
    PRIMARY=...), com fallback para a variavel de ambiente COMTRADE_KEY.
    A chave fica FORA do codigo de proposito -- nao versione o chave_api.txt.
    """
    arq = Path(__file__).resolve().parent / "dados_reais" / "comtrade" / "chave_api.txt"
    if arq.exists():
        for linha in arq.read_text(encoding="utf-8").splitlines():
            linha = linha.strip()
            if linha.upper().startswith("PRIMARY="):
                return linha.split("=", 1)[1].strip()
    return os.environ.get("COMTRADE_KEY", "").strip()


SUBSCRIPTION_KEY = _carregar_chave()

REPORTER_BRASIL = "76"        # codigo M49 do Brasil
FLOW_EXPORT = "X"             # exportacoes

# 10 commodities analisadas (HS-6)
HS6 = [
    "120190",  # soja em graos
    "230400",  # farelo de soja
    "150710",  # oleo de soja bruto
    "020230",  # carne bovina congelada
    "020714",  # carne de frango congelada
    "090111",  # cafe nao torrado
    "170114",  # acucar de cana bruto
    "200911",  # suco de laranja congelado
    "100590",  # milho
    "520100",  # algodao
]

ANOS_PRINCIPAIS = list(range(2014, 2025))   # 2014..2024 -- janela do modelo
ANOS_HISTORICO = list(range(2000, 2025))    # 2000..2024 -- proxy 'nunca-exportou'
BAIXAR_HISTORICO = True

OUT_DIR = Path(__file__).resolve().parent / "dados_reais" / "comtrade"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
def baixar(anos, rotulo):
    """Baixa as exportacoes do Brasil para os anos dados, 1 chamada por ano."""
    partes = []
    for ano in anos:
        print(f"  [{rotulo}] {ano} ...", end=" ", flush=True)
        try:
            df = comtradeapicall.getFinalData(
                SUBSCRIPTION_KEY,
                typeCode="C",            # commodities
                freqCode="A",            # anual
                clCode="HS",             # classificacao HS
                period=str(ano),
                reporterCode=REPORTER_BRASIL,
                cmdCode=",".join(HS6),
                flowCode=FLOW_EXPORT,
                partnerCode=None,        # todos os parceiros
                partner2Code="0",
                customsCode="C00",
                motCode="0",
                maxRecords=100000,
                format_output="JSON",
                breakdownMode="classic",
                includeDesc=True,
            )
            n = 0 if df is None else len(df)
            print(f"{n} linhas")
            if df is not None and n > 0:
                partes.append(df)
        except Exception as exc:
            print(f"ERRO: {exc}")
        time.sleep(1)  # gentileza com a API
    if not partes:
        return pd.DataFrame()
    return pd.concat(partes, ignore_index=True)


def main():
    if not SUBSCRIPTION_KEY:
        raise SystemExit("Chave nao encontrada. Coloque-a em "
                         "dados_reais/comtrade/chave_api.txt (linha PRIMARY=...) "
                         "ou na variavel de ambiente COMTRADE_KEY.")

    print(">> Janela principal (modelo):")
    df_main = baixar(ANOS_PRINCIPAIS, "principal")
    if not df_main.empty:
        # colunas-chave; mantemos todas, mas garantimos as essenciais
        essenciais = ["reporterCode", "reporterISO", "partnerCode", "partnerISO",
                      "cmdCode", "cmdDesc", "period", "flowCode",
                      "primaryValue", "netWgt", "qty"]
        cols = [c for c in essenciais if c in df_main.columns]
        out = OUT_DIR / "comtrade_exportacoes_br.csv"
        df_main.to_csv(out, index=False)
        print(f"   salvo: {out}  ({len(df_main)} linhas, {len(df_main.columns)} colunas)")
        print(f"   colunas essenciais presentes: {cols}")
    else:
        print("   nenhum dado retornado -- confira a chave de API.")

    if BAIXAR_HISTORICO:
        print(">> Janela historica (proxy 'nunca-exportou' / barreira sanitaria):")
        df_hist = baixar(ANOS_HISTORICO, "historico")
        if not df_hist.empty:
            out = OUT_DIR / "comtrade_historico_2000_2024.csv"
            df_hist.to_csv(out, index=False)
            print(f"   salvo: {out}  ({len(df_hist)} linhas)")

    print("\nConcluido. Arquivos salvos em dados_reais/comtrade/.")


if __name__ == "__main__":
    main()
