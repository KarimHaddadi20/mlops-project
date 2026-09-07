# Churn Classifier with Pipelines + MLflow

Un projet MLOps de classification de churn client utilisant scikit-learn Pipelines et MLflow pour le tracking d'expériences.

## 📋 Description

Ce projet implémente un classifieur de churn (attrition client) avec :
- **scikit-learn Pipeline** + **ColumnTransformer** pour le preprocessing reproductible
- **GridSearchCV** pour l'optimisation des hyperparamètres
- **MLflow** pour le tracking des expériences, métriques, et artifacts
- **Model Registry** pour la gestion des versions de modèles

## 📁 Structure du Projet

```
mlops-project/
├── data/                    # Données (gitignored)
│   └── raw.csv             # Dataset Telco Customer Churn
├── configs/
│   └── config.yaml         # Configuration du modèle et features
├── src/
│   ├── __init__.py
│   ├── pipeline.py         # Construction du Pipeline ML
│   ├── train.py            # Script d'entraînement avec MLflow
│   ├── evaluate.py         # Évaluation et génération d'artifacts
│   └── utils.py            # Fonctions utilitaires
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py    # Tests unitaires
├── Makefile                # Commandes automatisées
├── requirements.txt        # Dépendances Python
├── .env.example           # Template variables d'environnement
├── .gitignore
└── README.md
```

## 🚀 Installation

### 1. Cloner le projet

```bash
git clone <repo-url>
cd mlops-project
```

### 2. Créer l'environnement virtuel

```bash
make init
```

Ou manuellement :

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Télécharger le dataset

Téléchargez le dataset [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) et placez-le dans `data/raw.csv`.

### 4. Configurer les variables d'environnement (optionnel)

```bash
cp .env.example .env
# Éditer .env si nécessaire
```

## 📊 Utilisation

### Entraînement

```bash
make train
```

Ou avec un nom d'expérience personnalisé :

```bash
make train EXP=mon-experience
```

### Évaluation

```bash
make evaluate
```

### Lancer l'interface MLflow

```bash
make mlflow
```

Puis ouvrir http://localhost:5000

### Lancer les tests

```bash
make test
```

### Linting

```bash
make lint
```

## ⚙️ Configuration

Le fichier `configs/config.yaml` contient :

```yaml
data:
  csv_path: data/raw.csv
  target: Churn
  test_size: 0.2
  random_state: 42

features:
  numeric:
    - tenure
    - MonthlyCharges
    - TotalCharges
  categorical:
    - gender
    - Contract
    # ... autres features

model:
  type: logreg  # ou "random_forest"
  params:
    C: [0.1, 1.0, 10.0]
    penalty: ["l2"]
    solver: ["lbfgs", "liblinear"]

cv:
  strategy: StratifiedKFold
  n_splits: 5
  scoring: roc_auc
```

## 📈 Métriques et Artifacts

Le projet track automatiquement :

**Métriques :**
- ROC-AUC (validation croisée et test)
- Accuracy, Precision, Recall, F1-Score

**Artifacts :**
- Courbe ROC
- Courbe Precision-Recall
- Matrice de confusion
- Fichier CSV des prédictions pour analyse d'erreurs

## 🧪 Tests

Les tests vérifient :
- Construction du preprocessor
- Création des modèles
- Pipeline complet (fit/predict)
- Gestion des valeurs manquantes

```bash
pytest tests/ -v
```

## 📦 Model Registry

Le modèle est automatiquement enregistré dans MLflow Model Registry sous le nom `ChurnClassifier`.

Pour charger un modèle :

```python
import mlflow

model = mlflow.sklearn.load_model("models:/ChurnClassifier/latest")
```

## 🐳 Docker (Optionnel)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/train.py", "--config", "configs/config.yaml"]
```

## 📚 Ressources

- [scikit-learn Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## 📝 Licence

MIT License
