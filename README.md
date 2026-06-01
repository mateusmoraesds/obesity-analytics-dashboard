📊 Dashboard de Classificação de Obesidade

Aplicação de Machine Learning para classificação multiclasses de obesidade com base em hábitos de vida e características físicas, desenvolvida como Tech Challenge da Pós-Tech em Data Analytics da FIAP.


🌐 Acesse a aplicação
🔗 obesity-dashboard — Cloud Run

📋 Sumário

Sobre o projeto
Funcionalidades
Modelo de Machine Learning
Arquitetura
Stack utilizada
Como executar localmente
Estrutura do projeto
Autor


Sobre o projeto
A obesidade é um dos maiores desafios de saúde pública da atualidade. Este projeto utiliza Machine Learning para classificar o nível de obesidade de uma pessoa com base em hábitos de vida, dados comportamentais e características físicas, sem depender diretamente de peso ou altura para evitar leakage no modelo.
O resultado é um dashboard interativo que permite ao usuário preencher um formulário, obter sua classificação em tempo real, visualizar as probabilidades por classe e receber recomendações personalizadas geradas por IA.

Funcionalidades

Formulário interativo com 16 variáveis de entrada
Predição em tempo real com modelo XGBoost hospedado no GCS
Gráfico de probabilidade por classe com Plotly
Métricas de confiança da previsão e IMC calculado
Interpretação personalizada do resultado com recomendações
Recomendações avançadas geradas por IA (Groq + LLaMA 3.3 70B)
Chat interativo com IA restrito ao contexto de saúde e obesidade


Modelo de Machine Learning
Dataset
Obesity Dataset — variáveis comportamentais e hábitos de vida.
Classes preditas
ClasseDescrição0Peso Insuficiente1Peso Normal2Obesidade Grau I3Obesidade Grau II4Obesidade Grau III5Sobrepeso Nível I6Sobrepeso Nível II
Feature Engineering
pythondf["sedentary"]       = df["FAF"] * df["TUE"]
df["food_risk"]       = df["FAVC"] + df["CAEC"] + df["CALC"]
df["health_index"]    = df["FCVC"] + df["FAF"] + df["CH2O"]
df["lifestyle_score"] = df["FAF"] + df["FCVC"] + df["CH2O"] - df["TUE"] - df["FAVC"]
Tratamento de Leakage
Features IMC, IMC_cat, Weight, Height e Weight_Age foram removidas após análise SHAP que identificou leakage no modelo.
Resultados
MétricaValorAcurácia96%Classes7Features23AlgoritmoXGBoost

Arquitetura
Usuário (Browser)
      │
      ▼
Cloud Run (Streamlit)
      │
      ├──► Cloud Storage (xgb_obesity_model.pkl)
      │
      └──► Groq API (LLaMA 3.3 70B)

Frontend: Streamlit hospedado no Google Cloud Run
Modelo: XGBoost salvo em gs://tech-challenge-4-obesity/models/xgb_obesity_model.pkl
IA: Groq API com llama-3.3-70b-versatile para recomendações e chat


Stack utilizada
CategoriaTecnologiaLinguagemPython 3.11MLXGBoost, Scikit-Learn, SHAPDashboardStreamlit, PlotlyCloudGoogle Cloud Run, Google Cloud StorageIA GenerativaGroq API, LLaMA 3.3 70BVersionamentoGitHub

Como executar localmente
Pré-requisitos

Python 3.11+
Conta GCP com acesso ao bucket tech-challenge-4-obesity
Chave da Groq API

Instalação
bash# Clone o repositório
git clone https://github.com/mateusmoraesds/obesity-analytics-dashboard.git
cd obesity-analytics-dashboard

# Crie e ative o ambiente virtual
python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # Linux/Mac

# Instale as dependências
pip install -r requirements.txt
Configuração
Crie um arquivo .env na raiz do projeto:
GROQ_API_KEY=sua_chave_groq_aqui
Configure as credenciais do GCP:
bashgcloud auth application-default login
Execução
bashstreamlit run app.py
Acesse em http://localhost:8501

Estrutura do projeto
obesity-analytics-dashboard/
│
├── app.py                  # Aplicação principal Streamlit
├── gcs.py                  # Carregamento do modelo do GCS
├── requirements.txt        # Dependências do projeto
├── Dockerfile              # Configuração para deploy no Cloud Run
├── .gitignore
├── .env.example            # Exemplo de variáveis de ambiente
│
└── utils/
    ├── prediction.py       # Lógica de predição
    └── preprocessing.py    # Pré-processamento dos dados

Autor
Mateus Moraes

LinkedIn: mateusmoraesds
GitHub: mateusmoraesds


Projeto desenvolvido para o Tech Challenge Pós-Tech Data Analytics/FIAP — Porto Alegre, Maio de 2026
