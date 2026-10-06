# Modelo gravitacional com inflação de zeros para as exportações agropecuárias brasileiras (2014–2024)

Material de reprodução do Trabalho de Conclusão de Curso apresentado ao MBA em Data Science e Analytics da Escola Superior de Agricultura “Luiz de Queiroz”, Universidade de São Paulo (USP/Esalq), em 2026.

- Autor: Lucas Ataíde Heyden
- Orientador: Prof. Dr. Henrique Raymundo Gioia

O repositório reúne o painel final, os dados de entrada e as rotinas em Python que reproduzem integralmente os resultados do trabalho: modelos de Poisson, Binomial Negativa (NB), Poisson e Binomial Negativa inflacionados de zeros (ZIP e ZINB), protocolo de 50 pontos de partida para o ZINB, estimador de pseudo-máxima verossimilhança de Poisson (PPML), sensibilidade à unidade de medida, tipologia de parceiros por K-Means e modelos multinível (GLMM logístico e HLM2).

## Conteúdo

| Arquivo ou pasta | Descrição |
|---|---|
| `painel_real_milhares.csv` | Painel final: 22.400 observações parceiro-commodity-ano (207 parceiros, 10 commodities, 2014–2024), com o valor exportado em milhares de US$ |
| `painel_real_milhoes.csv` | Mesmo painel com o valor exportado em milhões de US$ (análise de sensibilidade) |
| `painel_real_meta_*.json` | Metadados dos painéis, incluindo as médias usadas na centragem das variáveis contínuas |
| `dados_reais/` | Dados de entrada: UN Comtrade, CEPII Gravity, World Development Indicators e Pink Sheet |
| `montar_painel_real.py` | Integra as fontes e gera o painel em milhões de US$ |
| `montar_paineis_alt.py` | Gera o painel principal, em milhares de US$ |
| `analise_tcc_final.py` | Rotina completa de estimação; grava `resultados_finais.json` |
| `glmm_agq.py` | GLMM logístico com intercepto aleatório, por máxima verossimilhança com quadratura de Gauss-Hermite adaptativa (rotina própria, com validação em dados simulados ao executar o arquivo) |
| `verificar_otimo.py` | Verificação do máximo do ZINB (hessiana e perturbações aleatórias) |
| `figuras_tcc.py`, `figstyle.py` | Geração das figuras (pasta `figuras/`): `fig1` a `fig3` correspondem às Figuras 1 a 3 do trabalho, `fig5_kmeans.png` à Figura 4, e `fig4_coef_zinb.png` é complementar |
| `reproduz_apendice.py` | Executa os trechos de código do Apêndice A e compara os resultados com `resultados_finais.json` |
| `resultados_finais.json` | Resultados numéricos citados no texto e nas tabelas |
| `baixar_comtrade.py` | Download dos dados do UN Comtrade pela API (requer chave gratuita, não incluída) |

## Variáveis do painel

| Variável | Descrição |
|---|---|
| `pais` | Código ISO3 do parceiro |
| `commodity` | Commodity (código SH de seis dígitos indicado no trabalho) |
| `ano` | Ano da exportação |
| `valor_exportado` | Valor exportado, arredondado ao inteiro (milhares ou milhões de US$, conforme o arquivo) |
| `ln_pib_parc` | Logaritmo do PIB do parceiro, centrado na média |
| `ln_pib_pc_parc` | Logaritmo do PIB *per capita* do parceiro, centrado na média |
| `ln_dist` | Logaritmo da distância ao parceiro, centrado na média |
| `idioma_comum` | 1 se o português é idioma oficial do parceiro |
| `rta` | 1 se há acordo comercial regional vigente (codificação do CEPII) |
| `acordo_sanitario` | Histórico de exportação: 1 se o Brasil exportou o produto ao parceiro entre 2000 e 2013 |
| `cambio` | Variação acumulada, desde 2014, do logaritmo do câmbio bilateral do real, centrada na média |
| `preco_lag` | Logaritmo do preço internacional da commodity no ano anterior, centrado na média |

## Como reproduzir

Requer Python 3.11 e as bibliotecas de `requirements.txt`.

```bash
pip install -r requirements.txt
python montar_painel_real.py   # reconstrói painel_real_milhoes.csv a partir de dados_reais/
python montar_paineis_alt.py   # reconstrói painel_real_milhares.csv
python analise_tcc_final.py    # estimações; grava resultados_finais.json
python verificar_otimo.py      # verificação do máximo do ZINB
python figuras_tcc.py          # figuras
python reproduz_apendice.py    # trechos do Apêndice A, com conferência dos resultados
```

Os painéis já estão incluídos, de modo que as duas primeiras etapas são opcionais e apenas os reconstroem a partir dos dados de entrada.

## Fontes dos dados

- UN Comtrade: exportações brasileiras por parceiro e produto (SH de seis dígitos), 2000–2024. United Nations [UN]. 2026. UN Comtrade Database. Disponível em: <https://comtradeplus.un.org>.
- CEPII Gravity, versão V202211: pares com o Brasil como origem (`gravity_brasil.csv`, extraído da base completa). Conte, M.; Cotterlaz, P.; Mayer, T. 2022. The CEPII Gravity Database. CEPII Working Paper 2022-05.
- World Development Indicators: PIB (NY.GDP.MKTP.CD), PIB *per capita* (NY.GDP.PCAP.CD) e taxa de câmbio oficial (PA.NUS.FCRF). World Bank. 2026.
- Pink Sheet: preços internacionais anuais das commodities. World Bank. 2026.

Os dados de entrada permanecem sujeitos aos termos de uso das respectivas fontes.
