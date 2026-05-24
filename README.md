# Obesity Behavioral Analytics

Projeto de Machine Learning e Analytics focado na predição de obesidade com base em fatores comportamentais e hábitos de vida.

O projeto utiliza XGBoost, SHAP e Streamlit para construir um pipeline completo de análise, interpretabilidade e visualização dos dados.

---

# Objetivo

Desenvolver um modelo de classificação multiclasses capaz de identificar níveis de obesidade utilizando variáveis relacionadas a:

- alimentação
- atividade física
- consumo de água
- tempo de tela
- histórico familiar
- hábitos de vida

Além da construção do modelo, o projeto também teve como foco:

- detecção de data leakage
- interpretabilidade com SHAP
- engenharia de features
- arquitetura cloud utilizando Google Cloud Platform

---

# Tecnologias Utilizadas

- Python
- Pandas
- XGBoost
- SHAP
- Scikit-Learn
- Streamlit
- Google Cloud Storage
- BigQuery
- Plotly

---

# Dataset

O dataset contém informações relacionadas a:

- gênero
- idade
- hábitos alimentares
- atividade física
- hidratação
- consumo de álcool
- transporte utilizado
- histórico familiar de obesidade

Target:

- Insufficient Weight
- Normal Weight
- Overweight Level I
- Overweight Level II
- Obesity Type I
- Obesity Type II
- Obesity Type III

---

# Feature Engineering

Foram criadas features derivadas para aumentar a capacidade preditiva do modelo:

| Feature | Descrição |
|---|---|
| sedentary | Relação entre atividade física e tempo de tela |
| food_risk | Índice de risco alimentar |
| health_index | Índice de hábitos saudáveis |
| lifestyle_score | Score geral de estilo de vida |

---

# Data Leakage Detection

Durante a análise, foi identificado leakage no modelo inicial através do SHAP.

Features removidas:

- IMC
- IMC_cat
- Weight
- Height
- Weight_Age

Após remoção das variáveis antropométricas, o modelo passou a aprender padrões comportamentais reais.

---

# Resultados

## Modelo Inicial

- Accuracy: ~97%
- Forte dependência de variáveis antropométricas

## Modelo Final

- Accuracy: ~88%
- Maior interpretabilidade
- Melhor generalização
- Redução significativa de leakage

---

# Interpretabilidade com SHAP

O projeto utiliza SHAP para:

- feature importance
- interpretação global
- interpretação local
- identificação de leakage
- análise de impacto das variáveis

---

# Arquitetura do Projeto


Notebook → Feature Engineering → XGBoost → SHAP
                    ↓
             Google Cloud Storage
                    ↓
               Streamlit App
