# OncoDeepMind
OncoDeepMind is a deep learning-based framework for predicting cancer drug responses using biological and molecular data. The goal is to support precision oncology through AI-powered analysis of complex datasets such as gene expression profiles and SMILES representations of drug compounds.

## 🚀 Features
- Deep neural networks with multi-layer architectures
- Integration of molecular and cellular biological data
- Predictive modeling of drug efficacy (e.g., AUC values)
- Customizable preprocessing and encoding pipelines
- FastAPI backend and Next.js frontend (if applicable)

## 📁 Project Structure

```bash
/CancerDetectionCNNN
├── /backend
│   ├── /data
│   │   ├── Cell_Lines_Details.xlsx
│   │   ├── Compounds-annotation.csv
│   │   ├── GDSC_DATASET.csv
│   │   ├── GDSC2_dataset.csv
│   │   └── The_Cancer_data_1500.csv
│   ├── /programs
│   │   ├── drug_neural_network.py
│   │   ├── main.py
│   │   ├── model.py
│   │   └── risk_model.py
│   ├── /saved_models
│   │   ├── cancer_risk_model.pkl
│   │   ├── cancer_risk_scaler.pkl
│   │   ├── categorical_cols.pkl
│   │   ├── columns.pkl
│   │   ├── DrugResponseModel.pth
│   │   ├── x_scaler.pkl
│   │   └── y_scaler.pkl
│   └── requirements.txt
├── /frontend
│   ├── /static
│   │   ├── script.js
│   │   └── style.css
│   └── /templates
│       ├── about.html
│       ├── index.html
│       ├── predict.html
│       └── risk.html
├── .gitignore
├── LICENSE
└── README.md
```

## 📦 Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/CancerDetectionCNN.git
cd CancerDetectionCNN
pip install -r requirements.txt
```

# 🧠 Usage
- Training a Model
python oncodeepmind/training/train_model.py --config configs/default.yaml
- Predicting Drug Response
python oncodeepmind/prediction/predict_auc.py --input data/test_samples.csv
- Run API (FastAPI backend)
uvicorn app.main.backend:app --reload

# 🔬 Technologies
Python, PyTorch, HTML, CSS, JavaScript
FastAPI, Joblib
Scikit-learn, Pandas, Numpy

# 📊 Datasets
GDSC (Genomics of Drug Sensitivity in Cancer)
CCLE (Cancer Cell Line Encyclopedia)

# 📈 Results
Model	| Dataset	| MAE	| R² Score
DNN (5-layer)	| GDSC	| 0.09	| 0.83
GraphNet | GDSC |	0.07 | 0.87

# 🛠️ TODO
1. Integrate GPT-2 HuggingFace Transformer for result explination
2. Publish webpage and deploy using Vercel (frontend) and Render (backend)

# 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.

# 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.


