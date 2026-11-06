# ML Capstone Project
# Regression + Clustering : California Housing      -> data/housing.csv
# Classification          : EmpatheticDialogues     -> data/empathetic.csv

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    silhouette_score,
)

# ---------------- Settings ----------------
RANDOM_STATE = 42       # one place to control all randomness
SHOW = False            # set to True in Jupyter / Colab so the plots appear in the notebook
REG_TARGET = "median_house_value"   # continuous target (Regression)

# Emotion groups for the binary classification target (1 = positive, 0 = negative).
# Ambiguous emotions are left out: surprised, anticipating, sentimental, nostalgic.
POSITIVE = ["excited", "proud", "grateful", "impressed", "confident", "hopeful",
            "joyful", "prepared", "content", "caring", "trusting", "faithful"]
NEGATIVE = ["annoyed", "angry", "sad", "lonely", "afraid", "disgusted", "terrified",
            "anxious", "disappointed", "guilty", "furious", "jealous", "embarrassed",
            "devastated", "ashamed", "apprehensive"]

os.makedirs("results", exist_ok=True)
log_lines = []


# ---------------- Helper functions ----------------
def log(text=""):
    """Print a line and keep it for results/metrics.txt."""
    print(text)
    log_lines.append(str(text))


def finish(filename):
    """Save the current figure (and show it if SHOW is True)."""
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    if SHOW:
        plt.show()
    plt.close()


def clean(df):
    """Handle missing values, then One-Hot Encode text columns."""
    print("\nMissing values per column:")
    print(df.isnull().sum())
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if text_cols:
        df = pd.get_dummies(df, columns=text_cols, drop_first=True)
    return df


# ==============================================================
# REGRESSION: California Housing
# ==============================================================

# ---- Data Loading and Initial Inspection ----
housing = pd.read_csv("data/housing.csv")
log("===== HOUSING DATASET =====")
log(f"Shape: {housing.shape}")
print("\nColumn data types:")
print(housing.dtypes)
print(housing.head())
log(f"Regression target: {REG_TARGET}")

# ---- Data Cleaning and Preparation ----
housing = clean(housing)

# Outlier detection: box plots of the features
num_cols = housing.select_dtypes(include=np.number).columns.drop(REG_TARGET)
plt.figure(figsize=(14, 6))
sns.boxplot(data=housing[num_cols])
plt.xticks(rotation=45)
plt.ylabel("Value")
plt.title("Box plots of the numerical features (dots are outliers)")
finish("results/housing_boxplots.png")

# The target has a spike at its maximum value: every house worth more than this was recorded as 500001
plt.figure(figsize=(7, 4))
sns.histplot(housing[REG_TARGET], bins=50)
plt.xlabel("Median house value (USD)")
plt.ylabel("Number of districts")
plt.title("Distribution of house prices (note the spike at the maximum, 500001)")
finish("results/housing_target_distribution.png")

# Decision: remove the districts whose price was capped at the maximum, because the real
# price is unknown and it would mislead a linear model.
max_price = housing[REG_TARGET].max()
n_before = len(housing)
housing = housing[housing[REG_TARGET] < max_price].reset_index(drop=True)
log(f"Removed {n_before - len(housing)} districts with a capped price ({max_price:.0f}); "
    f"{len(housing)} remain")

# Correlation heatmap (lower triangle only, the upper half is a mirror image)
corr = housing.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(12, 9))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation between features (red = move together, blue = move opposite)")
finish("results/housing_correlation.png")

# Price map: every dot is one district
plt.figure(figsize=(8, 6))
sc = plt.scatter(housing["longitude"], housing["latitude"],
                 c=housing[REG_TARGET], cmap="plasma", s=6)
plt.colorbar(sc, label="Median house value (USD)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("House prices across California (each dot is one district)")
finish("results/housing_price_map.png")

# ---- Scaling and Splitting (80% train / 20% test) ----
X_reg = housing.drop(columns=[REG_TARGET])
y_reg = housing[REG_TARGET]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=RANDOM_STATE
)
scaler_reg = StandardScaler()
Xr_train = scaler_reg.fit_transform(Xr_train)
Xr_test = scaler_reg.transform(Xr_test)

# ---- Linear Regression ----
log("\n===== Linear Regression =====")
lin_reg = LinearRegression()
lin_reg.fit(Xr_train, yr_train)
yr_pred = lin_reg.predict(Xr_test)

mse = mean_squared_error(yr_test, yr_pred)
rmse = np.sqrt(mse)
r2 = r2_score(yr_test, yr_pred)
log(f"MSE:  {mse:.4f}")
log(f"RMSE: {rmse:.4f}")
log(f"R2:   {r2:.4f}")

# Predicted vs actual: the closer the dots are to the red line, the better the model
plt.figure(figsize=(6, 6))
plt.scatter(yr_test, yr_pred, s=5, alpha=0.3)
lims = [min(yr_test.min(), yr_pred.min()), max(yr_test.max(), yr_pred.max())]
plt.plot(lims, lims, color="red", label="Perfect prediction")
plt.xlabel("Actual price (USD)")
plt.ylabel("Predicted price (USD)")
plt.title("Predicted vs actual house prices")
plt.legend()
finish("results/regression_predicted_vs_actual.png")

# Which features push the price up or down (features are standardized, so bars are comparable)
coef = pd.Series(lin_reg.coef_, index=X_reg.columns).sort_values()
plt.figure(figsize=(8, 6))
plt.barh(coef.index, coef.values, color=["tab:red" if v < 0 else "tab:green" for v in coef.values])
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Effect on price (green = raises it, red = lowers it)")
plt.title("What drives house prices in the model")
finish("results/regression_feature_effects.png")

# ==============================================================
# CLASSIFICATION: EmpatheticDialogues (text -> positive / negative)
# ==============================================================

# ---- Data Loading and Initial Inspection ----
emp = pd.read_csv("data/empathetic.csv")
log("\n\n===== EMPATHETIC DIALOGUES DATASET =====")
log(f"Shape: {emp.shape}")
print("\nColumn data types:")
print(emp.dtypes)
print(emp.head())

# ---- Data Cleaning and Preparation ----
print("\nMissing values per column:")
print(emp.isnull().sum())
emp = emp.dropna(subset=["situation", "emotion"])
emp_all = emp.copy()

# How the 32 emotions were grouped to build the binary target
counts = emp_all["emotion"].value_counts()
bar_colors = ["tab:green" if e in POSITIVE else "tab:red" if e in NEGATIVE else "tab:gray"
              for e in counts.index]
plt.figure(figsize=(8, 9))
plt.barh(counts.index[::-1], counts.values[::-1], color=bar_colors[::-1])
plt.xlabel("Number of conversations")
plt.title("The 32 emotions and how they were grouped")
plt.legend(handles=[Patch(color="tab:green", label="Positive"),
                    Patch(color="tab:red", label="Negative"),
                    Patch(color="tab:gray", label="Left out (ambiguous)")])
finish("results/empathetic_emotion_groups.png")

# Build the binary target: 1 = positive emotion, 0 = negative emotion
emp = emp[emp["emotion"].isin(POSITIVE + NEGATIVE)].copy()
emp["label"] = emp["emotion"].isin(POSITIVE).astype(int)
CLF_TARGET = "label"
log(f"Classification target: {CLF_TARGET} (1 = positive, 0 = negative)")
log(emp[CLF_TARGET].value_counts().to_string())

# EDA: text length
plt.figure(figsize=(7, 4))
sns.histplot(emp["situation"].str.split().str.len(), bins=40)
plt.xlabel("Words per situation")
plt.ylabel("Number of conversations")
plt.title("Length of the situation texts")
finish("results/empathetic_text_length.png")

# ---- Splitting (80% train / 20% test) and text vectorization ----
# TF-IDF turns the text into numbers (it also normalizes each row, so no StandardScaler is needed)
Xc_text_train, Xc_text_test, yc_train, yc_test = train_test_split(
    emp["situation"], emp[CLF_TARGET], test_size=0.2, random_state=RANDOM_STATE,
    stratify=emp[CLF_TARGET]
)
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
Xc_train = tfidf.fit_transform(Xc_text_train)
Xc_test = tfidf.transform(Xc_text_test)

# ---- Logistic Regression ----
log("\n===== Logistic Regression =====")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(Xc_train, yc_train)
yc_pred = log_reg.predict(Xc_test)

acc = accuracy_score(yc_test, yc_pred)
cm = confusion_matrix(yc_test, yc_pred)
tn, fp, fn, tp = cm.ravel()
prec = precision_score(yc_test, yc_pred)
rec = recall_score(yc_test, yc_pred)

log(f"Accuracy:  {acc:.4f}")
log(f"Confusion Matrix:\n{cm}")
log(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
log(f"Precision: {prec:.4f}")
log(f"Recall:    {rec:.4f}")

# Confusion matrix with names and percentages (percent of each actual class)
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
annot = np.array([[f"{cm[i, j]}\n{cm_pct[i, j]:.1f}%" for j in range(2)] for i in range(2)])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
            xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted emotion")
plt.ylabel("Actual emotion")
plt.title("Confusion Matrix")
finish("results/confusion_matrix.png")

# Words that push the prediction toward negative (red) or positive (green)
words = tfidf.get_feature_names_out()
coefs = log_reg.coef_[0]
order = np.argsort(coefs)
idx = np.concatenate([order[:10], order[-10:]])
plt.figure(figsize=(7, 7))
plt.barh(words[idx], coefs[idx], color=["tab:red"] * 10 + ["tab:green"] * 10)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Weight in the model")
plt.title("Words the model uses to tell negative from positive")
finish("results/classification_top_words.png")

# ==============================================================
# CLUSTERING: K-Means on California Housing (without the label)
# ==============================================================
log("\n===== K-Means Clustering =====")
X_cluster = housing.drop(columns=[REG_TARGET])
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

# Elbow Method: Inertia vs K
k_values = list(range(1, 11))
inertias = []
for k in k_values:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km.fit(X_cluster_scaled)
    inertias.append(km.inertia_)

# Pick the elbow: the point farthest from the line joining the first and last points
p1 = np.array([k_values[0], inertias[0]])
p2 = np.array([k_values[-1], inertias[-1]])
distances = []
for k, inertia in zip(k_values, inertias):
    p = np.array([k, inertia])
    d = p2 - p1
    v = p1 - p
    distances.append(abs(d[0] * v[1] - d[1] * v[0]) / np.linalg.norm(d))
optimal_k = k_values[int(np.argmax(distances))]
log(f"Optimal K (Elbow): {optimal_k}")

plt.figure(figsize=(6, 4))
plt.plot(k_values, inertias, marker="o")
plt.axvline(optimal_k, color="red", linestyle="--", label=f"Chosen K = {optimal_k}")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia (lower = tighter clusters)")
plt.title("Elbow Method")
plt.legend()
finish("results/elbow_plot.png")

# Fit final model
kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
cluster_labels = kmeans.fit_predict(X_cluster_scaled)

sil = silhouette_score(X_cluster_scaled, cluster_labels)
log(f"Silhouette Score: {sil:.4f}")

# Visualize clusters using two key features
plt.figure(figsize=(8, 6))
for c in range(optimal_k):
    members = cluster_labels == c
    plt.scatter(X_cluster.loc[members, "longitude"], X_cluster.loc[members, "latitude"],
                s=5, label=f"Cluster {c}")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"K-Means clusters on the map (K = {optimal_k})")
plt.legend(markerscale=3)
finish("results/clusters.png")

# Describe each cluster: average of a few key features
cluster_summary = (
    housing.assign(cluster=cluster_labels)
    .groupby("cluster")[["median_income", REG_TARGET, "housing_median_age", "population"]]
    .mean()
    .round(1)
)
cluster_summary["districts"] = pd.Series(cluster_labels).value_counts().sort_index()
cluster_summary.to_csv("results/cluster_summary.csv")
log("\nCluster summary (average per cluster):")
log(cluster_summary.to_string())

# ---------------- Save all numbers ----------------
with open("results/metrics.txt", "w") as f:
    f.write("\n".join(log_lines))
print("\nAll numbers were saved to results/metrics.txt")
