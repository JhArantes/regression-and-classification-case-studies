<div align="center">

# Regressão Linear & Logística
### Projetos supervisionados de Machine Learning com dados reais

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Concluído-22c55e?style=for-the-badge)

<br/>

> Quatro notebooks cobrindo dois problemas de **regressão** e dois de **classificação binária** —  
> desde a limpeza dos dados até a avaliação final dos modelos.

</div>

---

## 📂 Estrutura

```
📦 projeto
 ┣ 📓 Ex1-Imoveis-RegLinear.ipynb
 ┣ 📓 Ex2-ConsumoEnergia-Reglinear.ipynb
 ┣ 📓 Ex3-Classificacao-Binaria-RegLogistica.ipynb
 ┣ 📓 Ex4-Fraude-Ecommerce-RegLogistica.ipynb
 ┣ 📄 aluguel_imoveis.csv
 ┣ 📄 consumo_energia.csv
 ┣ 📄 evasao_academia.csv
 ┣ 📄 fraude_ecommerce.csv
 ┗ 📑 Exercícios para treino_ Regressão Linear e Logística.pdf
```

---

## 🗂 Visão Geral dos Projetos

| # | Notebook | Dataset | Problema | Modelo |
|:-:|---|---|---|---|
| 1 | `Ex1-Imoveis-RegLinear` | `aluguel_imoveis.csv` | Previsão de aluguel | Regressão Linear |
| 2 | `Ex2-ConsumoEnergia-Reglinear` | `consumo_energia.csv` | Consumo mensal de energia | Linear + Ridge |
| 3 | `Ex3-Classificacao-Binaria-RegLogistica` | `evasao_academia.csv` | Evasão de clientes | Regressão Logística |
| 4 | `Ex4-Fraude-Ecommerce-RegLogistica` | `fraude_ecommerce.csv` | Detecção de fraude | Regressão Logística |

---

## 📘 Ex1 — Previsão de Aluguel de Imóveis

**Target:** valor do aluguel &nbsp;|&nbsp; **Tipo:** Regressão

Previsão do preço de aluguel com base em características estruturais do imóvel como área, número de quartos e localização. Serve como baseline para comparação com modelos mais complexos.

**Métricas de avaliação:** MAE · RMSE · R²

---

## ⚡ Ex2 — Previsão de Consumo de Energia

**Target:** `consumo_kwh` &nbsp;|&nbsp; **Tipo:** Regressão

Uma distribuidora de energia precisa prever o consumo elétrico mensal de residências com base no perfil do imóvel e dos moradores.

<details>
<summary><b>🔍 Problemas encontrados na base</b></summary>

<br/>

| Inconsistência | Tratamento aplicado |
|---|---|
| `consumo_kwh` negativo ou nulo | Remoção das linhas |
| Valores ausentes no target | `dropna(subset=["consumo_kwh"])` |
| `moradores <= 0` | Convertidos para `NaN` |
| `area_m2 <= 0` | Convertidos para `NaN` |
| `renda_familiar <= 0` | Convertidos para `NaN` |
| `qtd_eletrodomesticos > 50` | Cap em 50 |
| `banhos_dia > 10` | Cap em 10 |

</details>

<details>
<summary><b>⚙️ Pipeline de pré-processamento</b></summary>

<br/>

```python
ColumnTransformer([
    ("num", SimpleImputer(strategy="median"),          num_features),
    ("cat", Pipeline([
        SimpleImputer(fill_value="Desconhecido"),
        OneHotEncoder(handle_unknown="ignore")
    ]),                                                cat_features)
])
```

</details>

<details>
<summary><b>✂️ Holdout & Modelos</b></summary>

<br/>

```python
train_test_split(test_size=0.25, random_state=42)
# 75% treino  |  25% teste
```

Dois modelos comparados em sequência:

| Modelo | R² | Observação |
|---|:-:|---|
| Regressão Linear | ~0.78 | Baseline sólido |
| Ridge Regression (`α=1.0`) | ~0.78 | Regularização sem perda significativa |

> **R² ≈ 0.78** — resultado satisfatório para modelo linear. Variáveis como número de moradores e presença de ar-condicionado apresentaram forte influência no consumo.

</details>

---

## 🏋️ Ex3 — Evasão de Clientes de Academia

**Target:** `cancelou` (0 = permaneceu · 1 = cancelou) &nbsp;|&nbsp; **Tipo:** Classificação Binária

Uma rede de academias quer identificar com antecedência quais clientes têm maior risco de cancelar o plano — permitindo ações de retenção antes da saída.

<details>
<summary><b>🔍 Problemas encontrados na base</b></summary>

<br/>

| Inconsistência | Tratamento aplicado |
|---|---|
| Valores ausentes no target | `dropna(subset=["cancelou"])` |
| `idade <= 0` | Convertidos para `NaN` |
| `frequencia_semanal > 14` | Cap em 14 |
| `tempo_cliente_meses <= 0` | Convertidos para `NaN` |

</details>

<details>
<summary><b>⚙️ Pipeline de pré-processamento</b></summary>

<br/>

Mesmo padrão do Ex2:

```python
ColumnTransformer([
    ("num", SimpleImputer(strategy="median"),          num_features),
    ("cat", Pipeline([
        SimpleImputer(fill_value="Desconhecido"),
        OneHotEncoder(handle_unknown="ignore")
    ]),                                                cat_features)
])
```

</details>

<details>
<summary><b>✂️ Holdout estratificado</b></summary>

<br/>

```python
train_test_split(test_size=0.25, random_state=42, stratify=y)
# 75% treino  |  25% teste  |  estratificado por classe
```

> A estratificação garante que a proporção de cancelamentos seja mantida em ambos os conjuntos — essencial para bases desbalanceadas.

</details>

<details>
<summary><b>⚖️ Desbalanceamento de classes & GridSearch</b></summary>

<br/>

O desbalanceamento foi tratado diretamente no estimador, sem oversampling:

```python
LogisticRegression(class_weight="balanced", max_iter=1000)
```

Hiperparâmetros otimizados via `GridSearchCV` com **5-fold cross-validation**:

```python
param_grid = {
    "classifier__C":      [0.01, 0.1, 1, 10],
    "classifier__solver": ["liblinear", "lbfgs"]
}

GridSearchCV(pipeline, param_grid, scoring="recall", cv=5, n_jobs=-1)
```

**Métrica de otimização: `recall`** — a prioridade do negócio é capturar clientes em risco, tornando os falsos negativos o erro mais custoso.

</details>

<details>
<summary><b>📊 Avaliação do modelo</b></summary>

<br/>

- Matriz de Confusão
- Classification Report (Precision · Recall · F1-Score por classe)

</details>

---

## 💳 Ex4 — Detecção de Fraude em E-commerce

**Target:** indicador de transação fraudulenta &nbsp;|&nbsp; **Tipo:** Classificação Binária

Identificação de transações fraudulentas em plataforma de e-commerce. Dado o forte desbalanceamento esperado (fraudes são eventos raros), o modelo é orientado a maximizar a detecção dos positivos.

---

## 🛠 Stack

<div align="center">

| Biblioteca | Finalidade |
|---|---|
| `pandas` | Leitura, limpeza e manipulação |
| `numpy` | Operações numéricas e cap de outliers |
| `sklearn.pipeline` | Encadeamento transformações + modelo |
| `sklearn.compose` | `ColumnTransformer` para features heterogêneas |
| `sklearn.impute` | Imputação de valores ausentes |
| `sklearn.model_selection` | Holdout, `GridSearchCV`, cross-validation |
| `sklearn.linear_model` | `LinearRegression`, `Ridge`, `LogisticRegression` |
| `sklearn.metrics` | MAE, RMSE, R², Recall, F1, Confusion Matrix |

</div>

---

## 📐 Fluxo Metodológico

```
Leitura dos dados
       │
       ▼
Limpeza ──────────────────────────────────────────────────────────────────
│  · Remoção de registros sem target                                      │
│  · Valores impossíveis → NaN (idades, áreas, rendas negativas)         │
│  · Cap de outliers (eletrodomésticos, frequência, banhos)              │
──────────────────────────────────────────────────────────────────────────
       │
       ▼
Separação X / y
       │
       ▼
Holdout ──── train_test_split  (estratificado quando classificação)
       │
       ▼
Pipeline ─── Imputer → Encoder → Modelo
       │
       ▼
Avaliação
  Regressão    : MAE · RMSE · R²
  Classificação: Recall · F1 · Confusion Matrix
```

---

## ▶️ Como executar

```bash
# 1. Clone o repositório
git clone https://github.com/JhArantes/regression-and-classification-case-studies
cd regression-and-classification-case-studies

# 2. Instale as dependências
pip install pandas numpy scikit-learn jupyter

# 3. Abra os notebooks
jupyter notebook
```

> ⚠️ Mantenha os arquivos `.csv` na mesma pasta dos notebooks.

---

## 📑 Referência

Baseado nos exercícios do documento **"Exercícios para treino: Regressão Linear e Logística"**.

---

<div align="center">
<sub>Desenvolvido com Python · scikit-learn · Jupyter</sub>
</div>