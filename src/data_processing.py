"""
Task 3 - Feature Engineering Pipeline

This module transforms raw transaction-level data into
customer-level model-ready features.

The pipeline performs:

1. Aggregate feature creation
2. Datetime feature extraction
3. Missing value handling
4. Categorical encoding
5. Feature scaling
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

def create_aggregate_features(df):
    """
    Create customer-level aggregate transaction features.
    """

    agg_df = (
        df.groupby("CustomerId")
        .agg(
            TotalTransactionAmount=("Amount", "sum"),
            AverageTransactionAmount=("Amount", "mean"),
            TransactionCount=("TransactionId", "count"),
            StdTransactionAmount=("Amount", "std")
        )
        .reset_index()
    )

    return agg_df



def extract_datetime_features(df):
    """
    Extract useful datetime features.
    """

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    df["TransactionHour"] = (
        df["TransactionStartTime"].dt.hour
    )

    df["TransactionDay"] = (
        df["TransactionStartTime"].dt.day
    )

    df["TransactionMonth"] = (
        df["TransactionStartTime"].dt.month
    )

    df["TransactionYear"] = (
        df["TransactionStartTime"].dt.year
    )

    return df


def drop_unused_columns(df):
    """
    Remove columns that are identifiers and
    do not contribute predictive information.
    """

    columns_to_drop = [
        "TransactionId",
        "BatchId",
        "AccountId",
        "SubscriptionId"
    ]

    df = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    return df



def get_feature_groups():
    """
    Define numerical and categorical features.
    """

    numerical_features = [
        "Amount",
        "Value",
        "PricingStrategy",
        "FraudResult",
        "TransactionHour",
        "TransactionDay",
        "TransactionMonth",
        "TransactionYear"
    ]

    categorical_features = [
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId",
        "CurrencyCode"
    ]

    return (
        numerical_features,
        categorical_features
    )

# Numerical Processing Pipeline:
# Numerical variables are processed using a two-step pipeline. First, missing values are replaced using median imputation. 
# Second, features are standardized using StandardScaler to ensure comparable scales across variables.
def create_numerical_pipeline():
    """
    Pipeline for numerical features.
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    return numerical_pipeline


def create_categorical_pipeline():
    """
    Pipeline for categorical features.
    """

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    return categorical_pipeline



def create_preprocessor():
    """
    Apply transformations to
    numerical and categorical features.
    """

    numerical_features, categorical_features = (
        get_feature_groups()
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                create_numerical_pipeline(),
                numerical_features
            ),

            (
                "cat",
                create_categorical_pipeline(),
                categorical_features
            )
        ]
    )

    return preprocessor


def build_pipeline():
    """
    Build complete preprocessing pipeline.
    """

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor()
            )
        ]
    )

    return pipeline


def prepare_data(df):
    """
    Complete feature engineering workflow.
    """

    df = drop_unused_columns(df)

    df = extract_datetime_features(df)

    return df



def transform_to_dataframe(df):
    """
    Return transformed dataframe.
    """

    df = prepare_data(df)

    pipeline = build_pipeline()

    transformed = pipeline.fit_transform(df)

    return pd.DataFrame(
        transformed.toarray()
        if hasattr(transformed, "toarray")
        else transformed
    )
