import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split
)

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import mlflow
import mlflow.sklearn


def load_data(path):

    return pd.read_csv(path)

def prepare_features(df):
    # 1. Base columns we always want to drop
    columns_to_drop = ["is_high_risk"]
    
    # 2. Dynamically scan for high-cardinality text columns (like IDs) before dummying
    # If a text column has more unique values than 50% of the dataset size, it's an ID.
    threshold = len(df) * 0.5 
    
    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
            unique_count = df[col].nunique()
            # If it's a unique identifier string, drop it entirely!
            if unique_count > threshold:
                print(f"⚠️ Dropping high-cardinality ID column: '{col}' ({unique_count} unique values)")
                columns_to_drop.append(col)
                
    # 3. Drop all target and ID columns safely
    # (using errors='ignore' in case some aren't present)
    X = df.drop(columns=columns_to_drop, errors='ignore')

    print(f"Shape of X before One-Hot Encoding: {X.shape}")

    # 4. Safely convert remaining low-cardinality categories (like 'Channel', 'CardType')
    X = pd.get_dummies(X, drop_first=True)
    
    print(f"Shape of X after One-Hot Encoding: {X.shape}")

    y = df["is_high_risk"]

    return X, y
    # Columns we absolutely cannot pass into the mathematical models
    columns_to_drop = ["is_high_risk"]
    
    # Check if ID columns exist in your dataset and add them to the drop list
    id_cols = ["TransactionId", "CustomerId", "Id"]
    for col in id_cols:
        if col in df.columns:
            columns_to_drop.append(col)
            
    # Drop target and metadata IDs
    X = df.drop(columns=columns_to_drop)

    # Convert remaining text/categorical columns into numeric dummy variables (One-Hot Encoding)
    # This transforms categories like "Product_A" or "Channel_Web" into 1s and 0s
    X = pd.get_dummies(X, drop_first=True)

    y = df["is_high_risk"]

    return X, y

def split_data(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(X_test)

    probabilities = (
        model.predict_proba(X_test)
        [:, 1]
    )

    metrics = {

        "accuracy":
        accuracy_score(
            y_test,
            predictions
        ),

        "precision":
        precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "recall":
        recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "f1":
        f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "roc_auc":
        roc_auc_score(
            y_test,
            probabilities
        )
    }

    return metrics

mlflow.set_experiment(
    "credit-risk-model"
)


def train_logistic_regression(
    X_train,
    y_train
):

    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    model.fit(
        X_train,
        y_train
    )

    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    return model

def log_model_run(
    model_name,
    model,
    metrics
):

    with mlflow.start_run(
        run_name=model_name
    ):

        mlflow.log_params(
            model.get_params()
        )

        mlflow.log_metrics(
            metrics
        )

        mlflow.sklearn.log_model(
            model,
            model_name
        )


def main():
    df = pd.read_csv("data/processed/processed_data.csv")

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    # Store the actual functions, not the evaluation results
    model_creators = {
        "LogisticRegression": lambda: train_logistic_regression(X_train, y_train),
        "RandomForest": lambda: train_random_forest(X_train, y_train)
    }

    for name, creator_func in model_creators.items():
        print(f"Starting MLflow run for {name}...")
        
        # Open the MLflow run context BEFORE evaluation or logging
        with mlflow.start_run(run_name=name):
            # 1. Train the model
            model = creator_func()
            
            # 2. Evaluate
            metrics = evaluate_model(model, X_test, y_test)
            print(name, metrics)

            # 3. Log everything directly inside the open run
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name)

if __name__ == "__main__":
    main()