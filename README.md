# OncoDeepMind
OncoDeepMind is a deep learning-based and AI-orientated web-application to help predict cancer drug responses using biological and molecular data. The goal is to support precision oncology through ML-powered analysis of complex datasets such as gene expression profiles and cell line pathways for prediciting the effectiveness of a drug's response certain cancers. OncoDeepMind also offers a risk predicition Logistic Regression model to provide users with probabilitiy of risk as well as generated recommendations for lifestyle changes to provide feedback to the user to lower risk and improve health.

## 🚀 Features
- Deep neural networks with multi-layer architectures
- Integration of molecular and cellular biological data
- Predictive modeling of drug efficacy (e.g., AUC values)
- Customizable preprocessing and encoding pipelines
- Feature weight system and custom lifestyle change recommendations
- FastAPI backend routing

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
│   └── /saved_models
│       ├── cancer_risk_model.pkl
│       ├── cancer_risk_scaler.pkl
│       ├── categorical_cols.pkl
│       ├── columns.pkl
│       ├── DrugResponseModel.pth
│       ├── x_scaler.pkl
│       └── y_scaler.pkl
├── /frontend
│   ├── /static
|   |   ├── /images
|   |   |   └── onco-logo-cropped.png
│   │   ├── script.js
│   │   └── style.css
│   └── /templates
│       ├── about.html
│       ├── index.html
│       ├── predict.html
│       └── risk.html
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## 📦 Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/OncoDeepMind-CancerDetectionCNN.git
cd CancerDetectionCNN
pip install -r requirements.txt
```

# 🧠 Usage
- Training + Testing Drug Response Deep Neural Network Model
python CancerDetectionCNN/backend/programs/drug_nueral_network.py
python CancerDetectionCNN/backend/programs/model.py
- Training Cancer Risk Logistic Regression Model
python CancerDetectionCNN/backend/programs/risk_model.py
- Run API (FastAPI backend)
uvicorn backend.programs.main:app --reload

# 🔬 Technologies
Python, HTML, CSS, JavaScript
PyTorch, FastAPI, Joblib
Scikit-learn, Pandas, Numpy

# 📊 Datasets
GDSC (Genomics of Drug Sensitivity in Cancer)
CCLE (Cancer Cell Line Encyclopedia)

# 📈 Results
| Model | Dataset | R² Score |
|----------|----------|----------|
| Deep Neural Network    | Data     | 0.66     |
| Logistic Regression    | The_Cancer_data_1500.csv     | 0.89     |

# 🛠️ TODO
1. Possible integration GPT-2 HuggingFace Transformer or Wrapper for future features
2. Publish webpage and deploy using Vercel (frontend) and Render (backend)

# ⚠️ Disclaimer
This is a personal project, and the predictions and suggestions provided by OncoDeepMind are based on machine learning models trained on historical data. These outputs are for informational purposes only and should not be considered as medical advice. Always consult with a licensed healthcare professional or oncologist before making any decisions related to cancer treatment or drug selection.

# 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.

# 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
