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
from sklearn.cluster import KMeans
from xverse.transformer import WOE

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


def create_rfm_features(df):
    """
    Create customer-level RFM metrics.
    """

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    snapshot_date = (
        df["TransactionStartTime"].max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=(
                "TransactionStartTime",
                lambda x:
                (
                    snapshot_date - x.max()
                ).days
            ),

            Frequency=(
                "TransactionId",
                "count"
            ),

            Monetary=(
                "Amount",
                "sum"
            )
        )
        .reset_index()
    )

    return rfm


def scale_rfm(rfm_df):

    scaler = StandardScaler()

    scaled_rfm = scaler.fit_transform(
        rfm_df[
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ]
    )

    return scaled_rfm


def cluster_customers(rfm_df):

    scaled_rfm = scale_rfm(rfm_df)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    rfm_df["Cluster"] = (
        kmeans.fit_predict(scaled_rfm)
    )

    return rfm_df


def identify_high_risk_cluster(clustered_df):
    """
    Automatically identify the least engaged cluster.
    """

    cluster_summary = (
        clustered_df
        .groupby("Cluster")
        [["Recency", "Frequency", "Monetary"]]
        .mean()
    )

    cluster_summary["RiskScore"] = (
        cluster_summary["Recency"]
        - cluster_summary["Frequency"]
        - cluster_summary["Monetary"]
    )

    high_risk_cluster = (
        cluster_summary["RiskScore"]
        .idxmax()
    )

    return high_risk_cluster


def create_target_variable(clustered_df):
    """
    Create binary is_high_risk target.
    """

    high_risk_cluster = (
        identify_high_risk_cluster(clustered_df)
    )

    clustered_df["is_high_risk"] = (
        clustered_df["Cluster"]
        == high_risk_cluster
    ).astype(int)

    return clustered_df

def merge_target_to_dataset(df):
    """
    Add is_high_risk target to
    transaction-level dataset.
    """

    rfm_df = create_rfm_features(df)

    clustered_df = cluster_customers(rfm_df)

    target_df = create_target_variable(
        clustered_df
    )

    final_df = df.merge(
        target_df[
            [
                "CustomerId",
                "is_high_risk"
            ]
        ],
        on="CustomerId",
        how="left"
    )

    return final_df

def apply_woe(df):
    """
    Apply Weight of Evidence transformation.
    """

    df = df.copy()

    target = "is_high_risk"

    categorical_cols = [
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId",
        "CurrencyCode"
    ]

    woe = WOE()

    woe.fit(
        df[categorical_cols],
        df[target]
    )

    transformed = woe.transform(
        df[categorical_cols]
    )

    return transformed

def generate_processed_dataset(df):
    """
    Complete Task 4 workflow.
    """

    final_df = merge_target_to_dataset(df)

    return final_df