﻿# Yelp Review Classification

This project involves building a machine learning pipeline to classify Yelp reviews based on their content. The pipeline includes data preprocessing, feature extraction, model training, and evaluation.

## Project Structure
```
yelp_review_classification/
├── data/
│   └── reviews.csv         
├── src/
│   ├── data_preprocessing.py  
│   ├── train_model.py         
│   ├── evaluate_model.py      
├── main.py                  
├── requirements.txt          
├── README.md                 
└── .gitignore
```
## Installation
1. Clone the repository:
```
git clone https://github.com/username/yelp-review-classification.git
cd yelp-review-classification
```
2. Install the required packages:
```
pip install -r requirements.txt
```
## Usage

Place your dataset (```reviews.csv```) in the data/ folder.

Run the ```main.py``` script to execute the pipeline:
```
python main.py
```

## Requirements

- Python 3.8+
- Required Python libraries (listed in ```requirements.txt```):
- numpy
- pandas
- scikit-learn

## Evaluation Metrics

The pipeline evaluates the model using the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score

