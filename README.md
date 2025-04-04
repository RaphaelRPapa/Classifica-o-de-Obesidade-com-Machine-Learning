# 🧠 Classificação de Obesidade com Machine Learning

Este projeto realiza a classificação do nível de obesidade de indivíduos com base em dados como IMC, hábitos alimentares, atividade física e outros fatores comportamentais. Utiliza algoritmos de aprendizado supervisionado, com destaque para o modelo Random Forest otimizado com Grid Search.

---

## 📁 Estrutura do Projeto

Todo o pipeline foi implementado em um único notebook:

- `classificacao_obesidade.ipynb` ✅

---

## 📊 Dataset

- **Nome:** ObesityDataSet_raw_and_data_sinthetic.csv  
- **Tarefa:** Classificação multiclasse  
- **Target:** `NObeyesdad` (nível de obesidade, com 7 categorias)  
- **Origem:** [Kaggle – Obesity Level Estimation](https://www.kaggle.com/datasets/mariaren/obesity-level-estimation)

---

## ⚙️ Pipeline do Projeto

### 1. 📊 Análise Exploratória (EDA)
- Distribuições de variáveis numéricas (IMC, Idade, Peso)
- Comparação entre Gênero e nível de obesidade
- Gráficos de dispersão, histogramas e mapas de calor de correlação

### 2. 🧹 Pré-Processamento
- Transformação de variáveis categóricas via `pd.get_dummies`
- Codificação da variável alvo (`LabelEncoder`)
- Normalização com `StandardScaler`
- Divisão entre treino e teste (80/20, estratificada)

### 3. 🤖 Modelagem
- Modelos testados:
  - Regressão Logística
  - Árvore de Decisão
  - KNN
  - SVM
  - ✅ **Random Forest** (modelo escolhido)

### 4. 🔧 Otimização (Tuning)
- Uso de `GridSearchCV` para encontrar os melhores hiperparâmetros do Random Forest:
  - `n_estimators`
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`
  - `max_features`

### 5. 📈 Avaliação
- Acurácia no conjunto de teste
- Matriz de confusão
- Relatório de classificação (Precisão, Recall, F1-score por classe)
- Comparação visual de modelos e métricas

---

## 💾 Outputs gerados

- `modelo_random_forest.pkl` – Modelo treinado e otimizado
- `scaler.pkl` – Escalonador de dados (caso reutilize o modelo)
- `matriz_confusao_rf_otimizado.png` – Matriz de confusão final
- `comparacao_modelos.png` – Gráfico comparativo entre modelos testados

---

## 🧪 Como Rodar o Projeto

### 1. Clone o repositório e instale as dependências:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
