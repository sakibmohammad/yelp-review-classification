from src.train_model import main as train_main
from src.evaluate_model import evaluate_model

if __name__ == "__main__":
    # Step 1: Prepare data for training
    X_train_fM, X_test_fM, y_train, y_test = train_main()

    # Step 2: Train a simple model (using Logistic Regression as an example)
    from sklearn.linear_model import LogisticRegression

    print("Training the model...")
    model = LogisticRegression()
    model.fit(X_train_fM, y_train)

    # Step 3: Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test_fM)

    # Step 4: Evaluate the model
    print("Evaluating the model...")
    metrics = evaluate_model(y_test, y_pred)

    # Output evaluation metrics
    print("\nFinal Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
