import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data_preprocessing import vocab_from_corpus, corpus_to_feature_matrix, vocab

# Load dataset
def load_data(filepath):
    """Loads data from a CSV file."""
    df = pd.read_csv(filepath)
    X = np.asarray(df['review'])
    y = np.asarray(df['rating'])
    return X, y

# Split dataset
def split_data(X, y, test_size=0.3):
    """Splits the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, shuffle=True)

# Prepare data for training
def prepare_training_data(X_train, X_test):
    """Builds vocabulary and feature matrices from the training and testing sets."""
    # Build vocabulary from training data
    vocab_from_corpus(X_train)
    
    # Get the vocabulary size
    N = len(vocab['t_2_i'])
    
    # Convert text data into feature matrices
    X_train_fM = corpus_to_feature_matrix(X_train, N)
    X_test_fM = corpus_to_feature_matrix(X_test, N)
    
    return X_train_fM, X_test_fM

# Main training workflow
def main():
    """Main function to load data, split, and prepare for model training."""
    # Filepath to the dataset
    filepath = 'data/reviews.csv'

    # Load data
    X, y = load_data(filepath)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Prepare feature matrices
    X_train_fM, X_test_fM = prepare_training_data(X_train, X_test)

    print("Training and testing data are ready for modeling.")
    return X_train_fM, X_test_fM, y_train, y_test

if __name__ == "__main__":
    main()
