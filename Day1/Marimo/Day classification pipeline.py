import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Import libraries and data
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import os
    import sys
    import re

    print(f"Pandas version: {pd.__version__}")
    print(f"Numpy version: {np.__version__}")

    # %% uncompress and read the data, and select the file Raw Data.csv from the zip #file_path = "c:\\Users\\emari\\OneDrive - Universitaet Bern\\GCB\\GRANTS\\DSL 2026 MURREN\\Course\\Day1"
    base_path = r"C:\Users\emari\OneDrive - Universitaet Bern\GCB\GRANTS\DSL 2026 MURREN\Course\Day1\Data"
    #base_path = r"C:\Users\emari\OneDrive - Universitaet Bern (1)\GCB\GRANTS\DSL 2026 MURREN\Course\Day1\Data"
    print(base_path)
    return base_path, os, pd


@app.cell
def _(os):
    print(os.path.exists(r"C:\Users\emari\OneDrive - Universitaet Bern\GCB\GRANTS\DSL 2026 MURREN\Course\Day1\Data\walk.zip"))
    print(os.path.exists(r"C:\Users\emari\OneDrive - Universitaet Bern\GCB\GRANTS\DSL 2026 MURREN\Course\Day1\Data\jump.zip"))

    return


@app.cell
def _(os, pd):
    def read_data(zip_name, base_path):
        import zipfile
        zip_path = os.path.join(base_path, zip_name)

        # Extract into a subfolder inside Day1
        extract_dir = os.path.join(base_path, "Data")
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

        # Load the CSV inside the extracted folder
        csv_path = os.path.join(extract_dir, "Raw Data.csv")
        df = pd.read_csv(csv_path)

        return df

    def clean_data(df):
        df = df.dropna()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        return df



    return (read_data,)


@app.cell
def _(base_path, read_data):
    walk_df = read_data(zip_name="walk.zip", base_path=base_path)
    jump_df = read_data(zip_name="jump.zip", base_path=base_path)

    #walk_df = clean_data(walk_df)
    #jump_df = clean_data(jump_df)


    walk_df.head(), jump_df.head()

    return jump_df, walk_df


@app.cell
def _(walk_df):
    walk_df.columns
    return


@app.cell
def _(mo, walk_df):

    column = mo.ui.dropdown(
        options=walk_df.columns.tolist(),
        value="Absolute acceleration (m/s^2)",
        label="Choose variable"
    )

    column
    return (column,)


@app.cell
def _(column, walk_df):
    import altair as alt

    alt.Chart(walk_df).mark_line().encode(
        x="Time (s):Q",
        y=alt.Y(column.value + ":Q", title=column.value))

    return (alt,)


@app.cell
def _(mo):
    axis_selector = mo.ui.multiselect(
        options=[
            'Linear Acceleration x (m/s^2)',
            'Linear Acceleration y (m/s^2)',
            'Linear Acceleration z (m/s^2)'
        ],
        value=['Linear Acceleration x (m/s^2)'],
        label="Select axes"
    )

    axis_selector

    return (axis_selector,)


@app.cell
def _(alt, axis_selector, walk_df):
    charts = [
        alt.Chart(walk_df).mark_line().encode(
            x="Time (s):Q",
            y=col + ":Q",
            color=alt.value(color)
        )
        for col, color in zip(axis_selector.value, ["red", "green", "blue"])
    ]

    charts[0] if len(charts) == 1 else alt.layer(*charts)

    return


@app.cell
def _(jump_df, pd, walk_df):
    walk_df["activity"] = "walk"
    jump_df["activity"] = "jump"

    movement = pd.concat([walk_df, jump_df])

    return (movement,)


@app.cell
def _(alt, movement):
    alt.Chart(movement).mark_line().encode(
        x="Time (s):Q",
        y="Absolute acceleration (m/s^2):Q",
        color="type:N")

    return


@app.cell
def _():
    # with mo.ui.tabs() as tabss:
    #     tabss.add("Table", walk_df)
    #     tabss.add("Plot", alt.Chart(walk_df).mark_line().encode(
    #         x="Time (s):Q",
    #         y="Absolute acceleration (m/s^2):Q"
    #     ))

    # tabss

    return


@app.cell
def _(alt, mo, movement):
    tabs = mo.ui.tabs({
        "Table": movement,
        "Plot": alt.Chart(movement).mark_line().encode(
            x="Time (s):Q",
            y="Absolute acceleration (m/s^2):Q"
        )
    })

    tabs

    return


@app.cell
def _(movement):
    # ============================================
    # 1. Encode labels (XGBoost requires numeric)
    # ============================================
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(movement["activity"])   # walk/jump → 0/1

    X = movement.drop(columns=["activity"])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    return X, X_test, X_train, le, y, y_test, y_train


@app.cell
def _(X_train):
    X_train.head()

    return


@app.cell
def _(X, X_test, X_train, le, y, y_test, y_train):
    # ============================================
    # 2. Define models with explanations
    # ============================================
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, classification_report

    models = {
        # Logistic Regression explanation
        "Logistic Regression": LogisticRegression(
            max_iter=500
            # Explanation:
            # A linear classifier that finds a separating hyperplane.
            # Interpretable coefficients, fast, good baseline.
        ),

        # Random Forest explanation
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42
            # Explanation:
            # Ensemble of decision trees.
            # Captures nonlinear patterns, robust to noise.
            # Provides feature importance via Gini impurity.
        ),

        # XGBoost explanation
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
            # Explanation:
            # Gradient-boosted trees.
            # Learns from mistakes of previous trees.
            # Often best performance on tabular data.
        ),

        # Neural Network explanation
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            max_iter=500,
            random_state=42
            # Explanation:
            # Learns nonlinear transformations.
            # Good for complex patterns.
            # Needs scaling and more data to shine.
        )
    }

    # ============================================
    # 3. Train and evaluate all models
    # ============================================
    results = {}
    predictions = {}

    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        predictions[name] = y_pred

        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ============================================
    # 4. Visualize model performance
    # ============================================
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8,5))
    plt.bar(results.keys(), results.values(),
            color=["steelblue","orange","green","purple"])
    plt.ylabel("Accuracy")
    plt.title("Model Performance Comparison")
    plt.ylim(0,1)
    plt.grid(axis="y", alpha=0.3)
    plt.show()

    # ============================================
    # 5. Feature importance for tree models
    # ============================================
    import pandas as pd

    # Random Forest
    rf = models["Random Forest"]
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

    plt.figure(figsize=(8,5))
    rf_importance.plot(kind="barh", color="steelblue")
    plt.title("Random Forest Feature Importance")
    plt.show()

    # XGBoost
    xgb = models["XGBoost"]
    xgb_importance = pd.Series(xgb.feature_importances_, index=X.columns).sort_values()

    plt.figure(figsize=(8,5))
    xgb_importance.plot(kind="barh", color="orange")
    plt.title("XGBoost Feature Importance")
    plt.show()

    # ============================================
    # 6. Logistic Regression coefficients
    # ============================================
    logreg = models["Logistic Regression"]
    coef = pd.Series(logreg.coef_[0], index=X.columns).sort_values()

    plt.figure(figsize=(8,5))
    coef.plot(kind="barh", color="green")
    plt.title("Logistic Regression Coefficients")
    plt.show()

    # ============================================
    # 7. Neural Network PCA visualization
    # ============================================
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8,5))
    plt.scatter(X_pca[:,0], X_pca[:,1],
                c=y, cmap="coolwarm", alpha=0.3)
    plt.title("Neural Network Input Space (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

    return pd, results


@app.cell
def _(results):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8,5))
    plt.bar(results.keys(), results.values(), color=["steelblue","orange","green","purple"])
    plt.ylabel("Accuracy")
    plt.title("Model Performance Comparison")
    plt.ylim(0,1)
    plt.grid(axis="y", alpha=0.3)
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
