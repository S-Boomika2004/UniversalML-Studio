import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Regression Models
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Clustering
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
)

from scipy.cluster.hierarchy import (
    dendrogram,
    linkage,
)

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Universal ML App",
    layout="wide"
)

st.title("Universal Machine Learning App")

# ==================================================
# FILE UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# ==================================================
# HELPER FUNCTIONS
# ==================================================

def remove_high_cardinality_columns(
    df,
    threshold=0.9
):
    """
    Remove ID-like columns
    """

    cols_to_drop = []

    for col in df.columns:

        unique_ratio = (
            df[col].nunique() / len(df)
        )

        if unique_ratio > threshold:
            cols_to_drop.append(col)

    df = df.drop(
        columns=cols_to_drop,
        errors="ignore"
    )

    return df, cols_to_drop


def detect_problem_type(y):
    """
    Detect classification or regression
    """

    if (
        y.dtype == "object"
        or y.nunique() <= 20
    ):
        return "Classification"

    return "Regression"


# ==================================================
# MAIN APP
# ==================================================

if uploaded_file:

    # ----------------------------------------------
    # READ DATASET
    # ----------------------------------------------

    try:

        df = pd.read_csv(uploaded_file)

    except Exception as e:

        st.error(f"Error reading file: {e}")
        st.stop()

    # ----------------------------------------------
    # DATA PREVIEW
    # ----------------------------------------------

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.write("Dataset Shape:", df.shape)

    # ----------------------------------------------
    # CLEAN DATA
    # ----------------------------------------------

    # Remove empty rows
    df.dropna(
        how="all",
        inplace=True
    )

    # Remove duplicates
    df.drop_duplicates(
        inplace=True
    )

    # Remove ID-like columns
    df, removed_cols = (
        remove_high_cardinality_columns(df)
    )

    if removed_cols:

        st.warning(
            f"Removed high-cardinality columns: "
            f"{removed_cols}"
        )

    # ==================================================
    # ML TYPE
    # ==================================================

    ml_type = st.selectbox(
        "Select ML Type",
        [
            "Classification",
            "Regression",
            "Clustering",
        ]
    )

    # ==================================================
    # CLUSTERING
    # ==================================================

    if ml_type == "Clustering":

        # Keep only numeric columns
        clustering_data = (
            df.select_dtypes(
                include=np.number
            )
        )

        # Validation
        if clustering_data.shape[1] < 2:

            st.error(
                "Need at least 2 numeric columns "
                "for clustering."
            )

            st.stop()

        # Fill missing values
        clustering_data = (
            clustering_data.fillna(
                clustering_data.mean()
            )
        )

        # Scaling
        scaler = StandardScaler()

        data_scaled = scaler.fit_transform(
            clustering_data
        )

        # Cluster Algorithm
        cluster_algo = st.selectbox(
            "Select Clustering Algorithm",
            [
                "K-Means",
                "Hierarchical",
            ]
        )

        # Safe max cluster value
        max_clusters = min(
            10,
            len(clustering_data) - 1
        )

        n_clusters = st.slider(
            "Number of Clusters",
            2,
            max_clusters,
            3
        )

        # ----------------------------------------------
        # RUN CLUSTERING
        # ----------------------------------------------

        if st.button("Run Clustering"):

            try:

                # ------------------------------
                # K-MEANS
                # ------------------------------

                if cluster_algo == "K-Means":

                    model = KMeans(
                        n_clusters=n_clusters,
                        random_state=42,
                        n_init=10
                    )

                    labels = model.fit_predict(
                        data_scaled
                    )

                # ------------------------------
                # HIERARCHICAL
                # ------------------------------

                else:

                    model = (
                        AgglomerativeClustering(
                            n_clusters=n_clusters
                        )
                    )

                    labels = model.fit_predict(
                        data_scaled
                    )

                    # Dendrogram
                    st.subheader("Dendrogram")

                    linked = linkage(
                        data_scaled,
                        method="ward"
                    )

                    fig, ax = plt.subplots(
                        figsize=(10, 5)
                    )

                    dendrogram(
                        linked,
                        ax=ax
                    )

                    st.pyplot(fig)

                # Add cluster labels
                clustering_result = (
                    clustering_data.copy()
                )

                clustering_result["Cluster"] = (
                    labels
                )

                st.subheader(
                    "Clustered Data"
                )

                st.dataframe(
                    clustering_result.head()
                )

                # Scatter Plot
                fig, ax = plt.subplots()

                ax.scatter(
                    data_scaled[:, 0],
                    data_scaled[:, 1],
                    c=labels
                )

                ax.set_xlabel(
                    clustering_data.columns[0]
                )

                ax.set_ylabel(
                    clustering_data.columns[1]
                )

                ax.set_title(
                    "Cluster Visualization"
                )

                st.pyplot(fig)

            except Exception as e:

                st.error(
                    f"Clustering Error: {e}"
                )

    # ==================================================
    # CLASSIFICATION + REGRESSION
    # ==================================================

    else:

        # ----------------------------------------------
        # TARGET SELECTION
        # ----------------------------------------------

        target = st.selectbox(
            "Select Target Column",
            df.columns
        )

        X = df.drop(columns=[target])

        y = df[target]

        # Remove rows where target is missing
        valid_index = y.dropna().index

        X = X.loc[valid_index]

        y = y.loc[valid_index]

        # ----------------------------------------------
        # FEATURE TYPES
        # ----------------------------------------------

        numeric_features = (
            X.select_dtypes(
                include=np.number
            ).columns
        )

        categorical_features = (
            X.select_dtypes(
                exclude=np.number
            ).columns
        )

        # ==================================================
        # PREPROCESSING
        # ==================================================

        numeric_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="mean"
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        categorical_transformer = Pipeline(
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

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numeric_transformer,
                    numeric_features
                ),
                (
                    "cat",
                    categorical_transformer,
                    categorical_features
                )
            ]
        )

        # ==================================================
        # CLASSIFICATION
        # ==================================================

        if ml_type == "Classification":

            # ----------------------------------------------
            # VALIDATION
            # ----------------------------------------------

            # Detect regression-like target
            if (
                pd.api.types.is_numeric_dtype(y)
                and y.nunique() > 20
            ):

                st.error(
                    "Selected target appears "
                    "continuous numeric.\n\n"
                    "Please use Regression."
                )

                st.stop()

            # Minimum classes
            if y.nunique() < 2:

                st.error(
                    "Target must contain "
                    "at least 2 classes."
                )

                st.stop()

            # Encode target
            if y.dtype == "object":

                y = pd.factorize(y)[0]

            # ----------------------------------------------
            # MODELS
            # ----------------------------------------------

            classification_models = {

                "Logistic Regression":
                    LogisticRegression(
                        max_iter=1000
                    ),

                "KNN":
                    KNeighborsClassifier(),

                "SVM":
                    SVC(),

                "Naive Bayes":
                    GaussianNB(),

                "Random Forest":
                    RandomForestClassifier(),

                "Decision Tree":
                    DecisionTreeClassifier(),
            }

            model_name = st.selectbox(
                "Select Algorithm",
                list(
                    classification_models.keys()
                )
            )

            model = (
                classification_models[
                    model_name
                ]
            )

            # ----------------------------------------------
            # SPLIT DATA
            # ----------------------------------------------

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42
                )
            )

            # ----------------------------------------------
            # TRAIN MODEL
            # ----------------------------------------------

            if st.button("Train Model"):

                try:

                    # --------------------------
                    # Naive Bayes
                    # --------------------------

                    if model_name == "Naive Bayes":

                        X_train_processed = (
                            preprocessor.fit_transform(
                                X_train
                            )
                        )

                        X_test_processed = (
                            preprocessor.transform(
                                X_test
                            )
                        )

                        # Convert sparse matrix
                        if hasattr(
                            X_train_processed,
                            "toarray"
                        ):

                            X_train_processed = (
                                X_train_processed.toarray()
                            )

                            X_test_processed = (
                                X_test_processed.toarray()
                            )

                        model.fit(
                            X_train_processed,
                            y_train
                        )

                        y_pred = model.predict(
                            X_test_processed
                        )

                    # --------------------------
                    # Other Models
                    # --------------------------

                    else:

                        pipeline = Pipeline(
                            steps=[
                                (
                                    "preprocessor",
                                    preprocessor
                                ),
                                (
                                    "model",
                                    model
                                )
                            ]
                        )

                        pipeline.fit(
                            X_train,
                            y_train
                        )

                        y_pred = pipeline.predict(
                            X_test
                        )

                    # --------------------------
                    # ACCURACY
                    # --------------------------

                    acc = accuracy_score(
                        y_test,
                        y_pred
                    )

                    st.success(
                        f"Accuracy: {acc:.4f}"
                    )

                    # --------------------------
                    # CONFUSION MATRIX
                    # --------------------------

                    cm = confusion_matrix(
                        y_test,
                        y_pred
                    )

                    fig, ax = plt.subplots()

                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        ax=ax
                    )

                    ax.set_title(
                        "Confusion Matrix"
                    )

                    st.pyplot(fig)

                except Exception as e:

                    st.error(
                        f"Classification Error: {e}"
                    )

        # ==================================================
        # REGRESSION
        # ==================================================

        else:

            # ----------------------------------------------
            # VALIDATION
            # ----------------------------------------------

            if not pd.api.types.is_numeric_dtype(y):

                st.error(
                    "Regression target "
                    "must be numeric."
                )

                st.stop()

            # ----------------------------------------------
            # MODELS
            # ----------------------------------------------

            regression_models = {

                "Linear Regression":
                    LinearRegression(),

                "Ridge":
                    Ridge(),

                "Lasso":
                    Lasso(),

                "Random Forest Regressor":
                    RandomForestRegressor(),

                "Decision Tree Regressor":
                    DecisionTreeRegressor(),
            }

            model_name = st.selectbox(
                "Select Algorithm",
                list(
                    regression_models.keys()
                )
            )

            model = (
                regression_models[
                    model_name
                ]
            )

            # ----------------------------------------------
            # SPLIT DATA
            # ----------------------------------------------

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42
                )
            )

            # ----------------------------------------------
            # TRAIN MODEL
            # ----------------------------------------------

            if st.button("Train Model"):

                try:

                    pipeline = Pipeline(
                        steps=[
                            (
                                "preprocessor",
                                preprocessor
                            ),
                            (
                                "model",
                                model
                            )
                        ]
                    )

                    pipeline.fit(
                        X_train,
                        y_train
                    )

                    y_pred = pipeline.predict(
                        X_test
                    )

                    # Metrics
                    mse = mean_squared_error(
                        y_test,
                        y_pred
                    )

                    r2 = r2_score(
                        y_test,
                        y_pred
                    )

                    st.success(
                        f"MSE: {mse:.4f}"
                    )

                    st.success(
                        f"R² Score: {r2:.4f}"
                    )

                    # Plot
                    fig, ax = plt.subplots()

                    ax.scatter(
                        y_test,
                        y_pred
                    )

                    ax.set_xlabel("Actual")

                    ax.set_ylabel(
                        "Predicted"
                    )

                    ax.set_title(
                        "Actual vs Predicted"
                    )

                    st.pyplot(fig)

                except Exception as e:

                    st.error(
                        f"Regression Error: {e}"
                    )

# ==================================================
# NO FILE
# ==================================================

else:

    st.info(
        "Please upload a CSV dataset."
    )
