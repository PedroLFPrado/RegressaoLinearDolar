# Regressão Linear do Preço do Dólar

## Descrição:

Este projeto implementa Regressão Linear para análise e previsão do preço do dólar no período de 28/10/2024 a 27/10/2025.

---

## Estrutura dos Arquivos

```
ProjFinal/
    ── Dolar.csv                               # Base de dados original (formato brasileiro)
    ── regressao_linear_dolar.py              # Script principal (CÓDIGO COMENTADO)
    ── README.md                               # Este arquivo (instruções)

    ── Arquivos Gerados (após execução):
        ── analise_regressao_linear_dolar.png      # Gráficos completos (6 visualizações)
        ── grafico_apresentacao_dolar.png          # Gráfico principal para apresentação
        ── resultados_regressao_linear.csv         # Dados + previsões + erros
        ── relatorio_analise.txt                   # Relatório textual detalhado
```

---

## Instalação das Dependências

### 1 - Verificar Python

Verifique se sua máquina possui Python instalado:

```powershell
python --version
```

### 2 - Instalar Bibliotecas Necessárias

Execute no PowerShell:

```powershell
pip install pandas numpy scikit-learn matplotlib reportlab
```

---

## Como Executar

```powershell
python regressao_linear_dolar.py
```

---

## Métricas de Desempenho

### 1. RMSE (Root Mean Squared Error)
- Erro médio em R$

### 2. MAE (Mean Absolute Error)
- Erro absoluto médio em R$
- Mais robusto a outliers

### 3. R² (Coeficiente de Determinação)
- Varia de 0 a 1 (0% a 100%)
- Indica % da variação explicada pelo modelo

### 4. MSE (Mean Squared Error)
- Penaliza erros grandes
- Base para cálculo do RMSE

---
