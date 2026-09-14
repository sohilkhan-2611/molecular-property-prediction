Molecular Property Prediction

Machine learning project for predicting molecular properties such as logP, aqueous solubility, and toxicity from molecular structures using data from PubChem.

Overview

The goal of this project is to build models that can look at a molecule’s chemical structure and predict an important molecular property.

The overall workflow is:

SMILES → Molecular Features → Machine Learning Model → Predicted Property

A molecule can be represented as a SMILES string, which is a text-based representation of its chemical structure. We convert these structures into numerical features and use them to train machine-learning models.

Objectives

* Collect molecular property data from PubChem
* Process molecular structures represented as SMILES
* Generate molecular features using RDKit
* Build traditional machine-learning baselines
* Build a Graph Neural Network (GNN)
* Compare different models
* Evaluate prediction performance

Models

The project will compare several approaches:

1. Random Forest

A simple and strong baseline for molecular property prediction.

2. XGBoost

A gradient-boosting model that can capture more complex relationships between molecular features and properties.

3. Graph Neural Network

A GNN represents a molecule as a graph:

* Atoms → Nodes
* Chemical bonds → Edges

Models such as GCN or MPNN can learn directly from the molecular structure.

Molecular Representations

Two main representations will be explored.

Morgan Fingerprints

Morgan fingerprints convert molecular structures into a vector describing the presence of different chemical patterns.

RDKit Descriptors

RDKit can calculate properties such as:

* Molecular weight
* Number of atoms
* Number of rings
* Hydrogen-bond donors
* Hydrogen-bond acceptors
* Lipophilicity-related descriptors

Dataset

The project uses molecular data from PubChem.

Example input:

CCO

This represents ethanol.

Example target:

logP = -0.3

The model learns from many molecules with known properties and then predicts the property of previously unseen molecules.

Project Structure

molecular-property-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_rdkit_features.ipynb
│   ├── 03_random_forest.ipynb
│   ├── 04_xgboost.ipynb
│   └── 05_gnn.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── models/
│
├── results/
│
├── requirements.txt
├── README.md
└── .gitignore

Evaluation

Since this is a regression problem, models will be evaluated using metrics such as:

* MAE — Mean Absolute Error
* RMSE — Root Mean Squared Error
* R² — Coefficient of Determination

The models will be compared to determine which approach gives the best predictions.

Technologies

* Python
* RDKit
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* PyTorch
* PyTorch Geometric
* PubChem

Learning Goals

This project provides hands-on experience with:

* Molecular data
* SMILES representations
* Chemical feature engineering
* Regression
* Random Forest
* XGBoost
* Graph Neural Networks
* Model evaluation
* Deep learning for molecular data
* Applying machine learning to drug-discovery problems

Example Workflow

                PubChem
                   │
                   ▼
             SMILES Dataset
                   │
                   ▼
             Data Cleaning
                   │
          ┌────────┴────────┐
          ▼                 ▼
   RDKit Features      Molecular Graph
          │                 │
          ▼                 ▼
 Random Forest/XGBoost     GNN
          │                 │
          └────────┬────────┘
                   ▼
            Model Comparison
                   │
                   ▼
          Molecular Property
              Prediction

Future Improvements

* Predict multiple molecular properties
* Experiment with different GNN architectures
* Hyperparameter optimization
* Interpret model predictions
* Add uncertainty estimation
* Deploy the best model as a simple prediction API or web application

License

This project is intended for educational and research purposes.
