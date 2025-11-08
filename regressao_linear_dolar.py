"""
=================================================================================
ANÁLISE DE DADOS - REGRESSÃO LINEAR PARA PREVISÃO DO PREÇO DO DÓLAR
=================================================================================

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar parâmetros de visualização
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


print("ANÁLISE DE REGRESSÃO LINEAR - PREÇO DO DÓLAR (28/10/2024 a 27/10/2025)")



# 1 -  CARREGAMENTO E PREPARAÇÃO DOS DADOS


print("\n1 - CARREGANDO E PREPARANDO OS DADOS:")

# Carregar o arquivo CSV
df = pd.read_csv('Dolar.csv', sep=';', decimal=',')

print(f"Dados carregados: {len(df)} registros")
print(f"Período: {df['Data'].iloc[-1]} até {df['Data'].iloc[0]}")

# Converter a coluna 'Data' para formato datetime
df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')

# Ordenar os dados por data (do mais antigo para o mais recente)
df = df.sort_values('Data').reset_index(drop=True)

# Criar variável numérica para representar o tempo (número de dias)
df['Dias'] = (df['Data'] - df['Data'].min()).dt.days

print(f"Dados ordenados cronologicamente")
print(f"Variável temporal criada (0 a {df['Dias'].max()} dias)")

# Visualizar primeiras e últimas linhas
print("\nPrimeiros registros:")
print(df[['Data', 'Valor', 'Dias']].head(3).to_string(index=False))
print("\nÚltimos registros:")
print(df[['Data', 'Valor', 'Dias']].tail(3).to_string(index=False))


# 2 - PREPARAÇÃO DAS VARIÁVEIS PARA O MODELO


print("\n2 - PREPARANDO VARIÁVEIS PARA O MODELO:")

# X: Dias
X = df[['Dias']].values

# y: Valor do dólar
y = df['Valor'].values

print(f"X (variável independente): shape = {X.shape}")
print(f"y (variável dependente): shape = {y.shape}")
print(f"Valor mínimo do dólar: R$ {y.min():.4f}")
print(f"Valor máximo do dólar: R$ {y.max():.4f}")
print(f"Valor médio do dólar: R$ {y.mean():.4f}")


# 3 - TREINAMENTO DO MODELO DE REGRESSÃO LINEAR


print("\n3 - TREINANDO O MODELO DE REGRESSÃO LINEAR:")

"""
- fit_intercept=True: Calcula o intercepto β₀ (padrão: True)
  Justificativa: Necessário para modelar corretamente o nível base do preço
  
- copy_X=True: Cria cópia dos dados de entrada (padrão: True)
  Justificativa: Evita modificações indesejadas nos dados originais
  
- n_jobs=None: Número de processadores para cálculos (padrão: None = 1)
  Justificativa: Para regressão simples, paralelização não é necessária
  
- positive=False: Não força coeficientes positivos (padrão: False)
  Justificativa: O coeficiente pode ser negativo se houver tendência de queda
"""

# Criar e treinar o modelo
modelo = LinearRegression(fit_intercept=True)
modelo.fit(X, y)

print("Modelo treinado com sucesso!")
print(f"\nPARÂMETROS ESTIMADOS:")
print(f"β₀ (Intercepto): {modelo.intercept_:.6f}")
print(f"β₁ (Coeficiente Angular): {modelo.coef_[0]:.6f}")

# Interpretar o coeficiente angular
if modelo.coef_[0] > 0:
    print(f"Tendência CRESCENTE: o dólar aumenta R$ {modelo.coef_[0]:.6f} por dia")
elif modelo.coef_[0] < 0:
    print(f"Tendência DECRESCENTE: o dólar diminui R$ {abs(modelo.coef_[0]):.6f} por dia")
else:
    print(f"Tendência ESTÁVEL: sem variação significativa")

# Equação da reta
print(f"\nEQUAÇÃO DA RETA:")
print(f"y = {modelo.intercept_:.6f} + {modelo.coef_[0]:.6f} × dias")


# 4 - FAZER PREVISÕES


print("\n4 - GERANDO PREVISÕES:")

# Fazer previsões para todos os pontos
y_pred = modelo.predict(X)
print(f"{len(y_pred)} previsões geradas")

# Comparar algumas previsões com valores reais
print("\nAmostra de previsões vs valores reais:")
print("   " + "-"*70)
print("   {:^12} | {:^15} | {:^15} | {:^15}".format(
    "Data", "Real (R$)", "Previsto (R$)", "Erro (R$)"))
print("   " + "-"*70)

# Mostrar primeiras 3, algumas do meio e últimas 3
indices_amostra = [0, 1, 2, len(df)//4, len(df)//2, 3*len(df)//4, 
                   len(df)-3, len(df)-2, len(df)-1]

for i in indices_amostra:
    data_str = df['Data'].iloc[i].strftime('%d/%m/%Y')
    real = y[i]
    previsto = y_pred[i]
    erro = real - previsto
    print("   {:^12} | {:^15.4f} | {:^15.4f} | {:^15.4f}".format(
        data_str, real, previsto, erro))


# 5 - AVALIAÇÃO DO DESEMPENHO DO MODELO



print("\n5 - ANÁLISE DE DESEMPENHO DO MODELO:")


# Calcular métricas de desempenho
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Calcular métricas adicionais
erro_percentual_medio = np.mean(np.abs((y - y_pred) / y)) * 100
variancia_real = np.var(y)
variancia_residuos = np.var(y - y_pred)

print("\nMÉTRICAS DE ERRO:")

print(f"\n1. MSE (Mean Squared Error - Erro Quadrático Médio):")
print(f"Valor: {mse:.6f}")
print(f"Interpretação: Média dos quadrados dos erros.")
print(f"Penaliza erros grandes. Quanto MENOR, melhor.")

print(f"\n2. RMSE (Root Mean Squared Error - Raiz do Erro Quadrático Médio):")
print(f"Valor: R$ {rmse:.4f}")
print(f"Interpretação: Desvio padrão dos erros na mesma unidade do dólar.")
print(f"Em média, o modelo erra ±R$ {rmse:.4f} na previsão.")

print(f"\n3. MAE (Mean Absolute Error - Erro Absoluto Médio):")
print(f"Valor: R$ {mae:.4f}")
print(f"Interpretação: Média dos erros absolutos.")
print(f"Em média, a diferença entre previsto e real é R$ {mae:.4f}.")

print(f"\n4. R² (Coeficiente de Determinação):")
print(f"Valor: {r2:.6f} ({r2*100:.2f}%)")
print(f"Interpretação: Proporção da variância explicada pelo modelo.")
if r2 > 0.9:
    print(f"EXCELENTE: O modelo explica {r2*100:.2f}% da variação dos dados.")
elif r2 > 0.7:
    print(f"BOM: O modelo explica {r2*100:.2f}% da variação dos dados.")
elif r2 > 0.5:
    print(f"MODERADO: O modelo explica {r2*100:.2f}% da variação dos dados.")
else:
    print(f"FRACO: O modelo explica apenas {r2*100:.2f}% da variação.")

print(f"\n5. ERRO PERCENTUAL MÉDIO:")
print(f"Valor: {erro_percentual_medio:.2f}%")
print(f"Interpretação: Erro médio em relação ao valor real.")

print(f"\n6. VARIÂNCIA DOS DADOS:")
print(f"Variância dos valores reais: {variancia_real:.6f}")
print(f"Variância dos resíduos: {variancia_residuos:.6f}")
print(f"Redução da variância: {((1 - variancia_residuos/variancia_real)*100):.2f}%")

# Análise dos resíduos
residuos = y - y_pred
print(f"\nANÁLISE DOS RESÍDUOS (ERROS):")
print(f"Média dos resíduos: {np.mean(residuos):.6f}")
print(f"(Deve estar próximo de 0 para um bom modelo)")
print(f"Desvio padrão dos resíduos: {np.std(residuos):.6f}")
print(f"Resíduo mínimo: {np.min(residuos):.6f}")
print(f"Resíduo máximo: {np.max(residuos):.6f}")


# ETAPA 6: GRÁFICOS


print("\n6 - GERANDO VISUALIZAÇÕES:")

# Criar figura com múltiplos subplots
fig = plt.figure(figsize=(16, 12))


# Gráfico 1: Série Temporal - Real vs Previsto
ax1 = plt.subplot(3, 2, 1)
ax1.plot(df['Data'], y, 'o-', color='#2E86AB', linewidth=2, 
         markersize=4, label='Preço Real', alpha=0.7)
ax1.plot(df['Data'], y_pred, '--', color='#E63946', linewidth=2.5, 
         label='Regressão Linear', alpha=0.8)
ax1.set_xlabel('Data', fontsize=11, fontweight='bold')
ax1.set_ylabel('Valor do Dólar (R$)', fontsize=11, fontweight='bold')
ax1.set_title('Preço do Dólar: Real vs Previsto pela Regressão Linear', 
              fontsize=12, fontweight='bold', pad=15)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.tick_params(axis='x', rotation=45)


# Gráfico 2: Dispersão - Real vs Previsto
ax2 = plt.subplot(3, 2, 2)
ax2.scatter(y, y_pred, alpha=0.6, color='#06A77D', s=30, edgecolors='black', linewidth=0.5)
# Linha de referência perfeita (y = x)
min_val, max_val = y.min(), y.max()
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
         label='Predição Perfeita', alpha=0.7)
ax2.set_xlabel('Valor Real (R$)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Valor Previsto (R$)', fontsize=11, fontweight='bold')
ax2.set_title(f'Correlação: Real vs Previsto (R² = {r2:.4f})', 
              fontsize=12, fontweight='bold', pad=15)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')


# Gráfico 3: Resíduos ao longo do tempo
ax3 = plt.subplot(3, 2, 3)
ax3.plot(df['Data'], residuos, 'o-', color='#F77F00', 
         markersize=4, linewidth=1, alpha=0.7)
ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Erro Zero')
ax3.fill_between(df['Data'], residuos, 0, alpha=0.3, color='#F77F00')
ax3.set_xlabel('Data', fontsize=11, fontweight='bold')
ax3.set_ylabel('Resíduo (Erro) R$', fontsize=11, fontweight='bold')
ax3.set_title('Resíduos (Erros) ao Longo do Tempo', 
              fontsize=12, fontweight='bold', pad=15)
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.tick_params(axis='x', rotation=45)


# Gráfico 4: Histograma dos Resíduos
ax4 = plt.subplot(3, 2, 4)
ax4.hist(residuos, bins=30, color='#9D4EDD', alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Erro Zero')
ax4.set_xlabel('Resíduo (R$)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequência', fontsize=11, fontweight='bold')
ax4.set_title('Distribuição dos Resíduos', 
              fontsize=12, fontweight='bold', pad=15)
ax4.legend(loc='best', fontsize=10)
ax4.grid(True, alpha=0.3, axis='y', linestyle='--')


# Gráfico 5: Erro Absoluto ao longo do tempo
ax5 = plt.subplot(3, 2, 5)
erro_abs = np.abs(residuos)
ax5.bar(df['Data'], erro_abs, color='#D62828', alpha=0.6, width=2)
ax5.axhline(y=mae, color='blue', linestyle='--', linewidth=2, 
            label=f'MAE = R$ {mae:.4f}')
ax5.set_xlabel('Data', fontsize=11, fontweight='bold')
ax5.set_ylabel('Erro Absoluto (R$)', fontsize=11, fontweight='bold')
ax5.set_title('Erro Absoluto das Previsões', 
              fontsize=12, fontweight='bold', pad=15)
ax5.legend(loc='best', fontsize=10)
ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
ax5.tick_params(axis='x', rotation=45)


# Gráfico 6: Métricas de Desempenho
ax6 = plt.subplot(3, 2, 6)
ax6.axis('off')

# Texto com métricas
metricas_texto = f"""
RESUMO DO DESEMPENHO DO MODELO
{'='*50}

Período Analisado:
  • Data Inicial: {df['Data'].min().strftime('%d/%m/%Y')}
  • Data Final: {df['Data'].max().strftime('%d/%m/%Y')}
  • Total de Dias: {len(df)} registros

Parâmetros do Modelo:
  • Intercepto (β₀): R$ {modelo.intercept_:.4f}
  • Coeficiente (β₁): {modelo.coef_[0]:.6f} R$/dia

Métricas de Erro:
  • RMSE: R$ {rmse:.4f}
  • MAE: R$ {mae:.4f}
  • Erro Percentual Médio: {erro_percentual_medio:.2f}%

Qualidade do Ajuste:
  • R² (Coeficiente de Determinação): {r2:.4f}
  • Variância Explicada: {r2*100:.2f}%

Estatísticas do Preço:
  • Preço Mínimo: R$ {y.min():.4f}
  • Preço Máximo: R$ {y.max():.4f}
  • Preço Médio: R$ {y.mean():.4f}
  • Desvio Padrão: R$ {np.std(y):.4f}
"""

ax6.text(0.1, 0.95, metricas_texto, transform=ax6.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Ajustar layout
plt.tight_layout()
plt.savefig('analise_regressao_linear_dolar.png', dpi=300, bbox_inches='tight')
print("Gráfico salvo como 'analise_regressao_linear_dolar.png'")

# Gráfico Adicional: Foco na Comparação Real vs Previsto (para apresentação)
fig2, ax = plt.subplots(figsize=(14, 7))
ax.plot(df['Data'], y, 'o-', color='#1E3A8A', linewidth=2.5, 
        markersize=5, label='Preço Real do Dólar', alpha=0.8)
ax.plot(df['Data'], y_pred, '--', color='#DC2626', linewidth=3, 
        label='Previsão (Regressão Linear)', alpha=0.9)
ax.fill_between(df['Data'], y, y_pred, alpha=0.2, color='gray')
ax.set_xlabel('Data', fontsize=13, fontweight='bold')
ax.set_ylabel('Valor do Dólar (R$)', fontsize=13, fontweight='bold')
ax.set_title('ANÁLISE DO DÓLAR: Preço Real vs Previsão por Regressão Linear\n' + 
             f'Período: 28/10/2024 a 27/10/2025 | R² = {r2:.4f} | RMSE = R$ {rmse:.4f}',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('grafico_apresentacao_dolar.png', dpi=300, bbox_inches='tight')
print("Gráfico para apresentação salvo como 'grafico_apresentacao_dolar.png'")


# ETAPA 7: SALVAR RESULTADOS 
print("\n7 - SALVANDO RESULTADOS:")

# Adicionar previsões ao DataFrame
df['Valor_Previsto'] = y_pred
df['Erro'] = residuos
df['Erro_Absoluto'] = erro_abs
df['Erro_Percentual'] = (erro_abs / y) * 100

# Salvar em CSV
df.to_csv('resultados_regressao_linear.csv', index=False, decimal=',', sep=';')
print("Resultados salvos em 'resultados_regressao_linear.csv'")

# Salvar relatório detalhado
with open('relatorio_analise.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("RELATÓRIO DE ANÁLISE - REGRESSÃO LINEAR DO PREÇO DO DÓLAR\n")
    f.write("="*80 + "\n\n")
    
    f.write("PERÍODO ANALISADO:\n")
    f.write(f"Data Inicial: {df['Data'].min().strftime('%d/%m/%Y')}\n")
    f.write(f"Data Final: {df['Data'].max().strftime('%d/%m/%Y')}\n")
    f.write(f"Total de Registros: {len(df)}\n\n")

    f.write("MODELO DE REGRESSÃO LINEAR:\n")
    f.write(f"Equação: y = {modelo.intercept_:.6f} + {modelo.coef_[0]:.6f} × dias\n")
    f.write(f"Intercepto (β₀): R$ {modelo.intercept_:.6f}\n")
    f.write(f"Coeficiente Angular (β₁): {modelo.coef_[0]:.6f} R$/dia\n\n")

    f.write("MÉTRICAS DE DESEMPENHO:\n")
    f.write(f"MSE (Erro Quadrático Médio): {mse:.6f}\n")
    f.write(f"RMSE (Raiz do EQM): R$ {rmse:.4f}\n")
    f.write(f"MAE (Erro Absoluto Médio): R$ {mae:.4f}\n")
    f.write(f"R² (Coeficiente de Determinação): {r2:.6f} ({r2*100:.2f}%)\n")
    f.write(f"Erro Percentual Médio: {erro_percentual_medio:.2f}%\n\n")

    f.write("ESTATÍSTICAS DO PREÇO DO DÓLAR:\n")
    f.write(f"Mínimo: R$ {y.min():.4f}\n")
    f.write(f"Máximo: R$ {y.max():.4f}\n")
    f.write(f"Média: R$ {y.mean():.4f}\n")
    f.write(f"Mediana: R$ {np.median(y):.4f}\n")
    f.write(f"Desvio Padrão: R$ {np.std(y):.4f}\n\n")
    
    f.write("INTERPRETAÇÃO:\n")
    f.write(f"O modelo de regressão linear {'' if r2 > 0.7 else 'não '}apresenta um bom ajuste\n")
    f.write(f"aos dados, explicando {r2*100:.2f}% da variação no preço do dólar.\n")
    f.write(f"O erro médio das previsões é de R$ {mae:.4f}, representando\n")
    f.write(f"aproximadamente {erro_percentual_medio:.2f}% do valor real.\n")

print("Relatório detalhado salvo em 'relatorio_analise.txt'")


# FINALIZAÇÃO
print("\n" + "-"*80)
print("ANÁLISE CONCLUÍDA COM SUCESSO!")

print("\nArquivos gerados:")
print("  1. analise_regressao_linear_dolar.png - Gráficos completos")
print("  2. grafico_apresentacao_dolar.png - Gráfico principal para apresentação")
print("  3. resultados_regressao_linear.csv - Dados com previsões")
print("  4. relatorio_analise.txt - Relatório detalhado")
print("\nTodos os arquivos foram salvos no diretório atual.")

# Mostrar os gráficos
plt.show()
