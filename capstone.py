# ML Capstone Project
# Regression + Clustering : NYC Airbnb            -> data/airbnb.csv
# Classification          : EmpatheticDialogues   -> data/empathetic.csv


import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# ---------------- Settings ----------------
RANDOM_STATE = 42                 # used for the main results
SEEDS = [42, 43, 44, 45, 46]      # the before/after experiments are repeated on 5 different splits
SHOW = False                      # True in a notebook, so the plots appear under the cells

BASE = "." if os.path.isdir("data") else (".." if os.path.isdir("../data") else ".")
DATA = f"{BASE}/data"
RES = f"{BASE}/results"
os.makedirs(DATA, exist_ok=True)
os.makedirs(RES, exist_ok=True)

# Emotion groups for the binary classification target (1 = positive, 0 = negative).
# Ambiguous emotions are left out: surprised, anticipating, sentimental, nostalgic.
POSITIVE = ["excited", "proud", "grateful", "impressed", "confident", "hopeful",
            "joyful", "prepared", "content", "caring", "trusting", "faithful"]
NEGATIVE = ["annoyed", "angry", "sad", "lonely", "afraid", "disgusted", "terrified",
            "anxious", "disappointed", "guilty", "furious", "jealous", "embarrassed",
            "devastated", "ashamed", "apprehensive"]

log_lines = []

# ==== Helper functions ====
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


def reg_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}


def clf_metrics(y_true, y_pred, y_score):
    return {"Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "ROC_AUC": roc_auc_score(y_true, y_score)}


def summarize(runs, metrics):
    """Mean and standard deviation of each metric over the splits, for every step."""
    table = runs.groupby("step")[metrics].agg(["mean", "std"])
    table.columns = [f"{m}_{s}" for m, s in table.columns]
    return table


def paired_deltas(runs, metrics, primary, lower_is_better):
    """Change from the previous step, measured on the same split, so the comparison is fair."""
    rows = []
    for s in sorted(runs["step"].unique())[1:]:
        prev = runs[runs["step"] == s - 1].set_index("seed")[metrics]
        cur = runs[runs["step"] == s].set_index("seed")[metrics]
        diff = cur - prev
        better = (diff[primary] < 0) if lower_is_better else (diff[primary] > 0)
        row = {"step": s}
        for m in metrics:
            row[m + "_change"] = diff[m].mean()
        row["better_in"] = f"{int(better.sum())} of {len(better)} splits"
        rows.append(row)
    return pd.DataFrame(rows).set_index("step")


def plot_steps(summary, metrics, prefix, filename, title):
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.8 * len(metrics), 4))
    for ax, m in zip(np.atleast_1d(axes), metrics):
        ax.errorbar(summary.index, summary[m + "_mean"], yerr=summary[m + "_std"],
                    marker="o", capsize=3)
        ax.set_xticks(summary.index)
        ax.set_xticklabels([f"{prefix}{i}" for i in summary.index])
        ax.set_xlabel("Cleaning step (each step adds to the previous ones)")
        ax.set_title(m)
    fig.suptitle(title)
    finish(filename)

# ==== Get the data ====
GITHUB_USER = "3MINA3"

if not os.path.exists(f"{DATA}/airbnb.csv"):
    import urllib.request
    url = f"https://raw.githubusercontent.com/{GITHUB_USER}/ML-Capstone-Project/main/data/airbnb.csv"
    urllib.request.urlretrieve(url, f"{DATA}/airbnb.csv")

if not os.path.exists(f"{DATA}/empathetic.csv"):
    from datasets import load_dataset
    ds = load_dataset("Estwld/empathetic_dialogues_llm")
    df_emp = pd.concat([split.to_pandas() for split in ds.values()])
    df_emp["first_message"] = df_emp["conversations"].apply(lambda c: c[0]["content"])
    df_emp[["emotion", "situation", "first_message"]].to_csv(f"{DATA}/empathetic.csv", index=False)

# ==== Part A: Regression with Linear Regression (NYC Airbnb) ====
air = pd.read_csv(f"{DATA}/airbnb.csv")
log("===== AIRBNB DATASET =====")
log(f"Shape: {air.shape}")
print("\nColumn data types:")
print(air.dtypes)
print(air.head())
log("Regression target: price (USD per night)")

missing = air.isnull().sum()
log("\nMissing values per column:")
log(missing[missing > 0].to_string())
log("\nPrice summary:")
log(air["price"].describe().round(2).to_string())
log(f"Listings with price = 0: {(air['price'] == 0).sum()}")
log(f"Duplicate rows: {air.duplicated().sum()}")

NUM = ["latitude", "longitude", "minimum_nights", "number_of_reviews",
    "reviews_per_month", "calculated_host_listings_count", "availability_365"]
CAT = ["neighbourhood_group", "room_type"]
CLIP_COLS = ["minimum_nights", "number_of_reviews", "reviews_per_month",
            "calculated_host_listings_count"]

# ==== 2. Exploratory data analysis ====
# Missing values
miss = air.isnull().sum().sort_values(ascending=False)
miss = miss[miss > 0]
plt.figure(figsize=(7, 3.5))
plt.barh(miss.index[::-1], miss.values[::-1])
plt.xlabel("Number of missing values")
plt.title("Missing values per column")
finish(f"{RES}/airbnb_missing_values.png")

# Price distribution before and after a log transform
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(air["price"], bins=60, ax=axes[0])
axes[0].set_yscale("log")
axes[0].set_xlabel("Price per night (USD)")
axes[0].set_ylabel("Number of listings (log scale)")
axes[0].set_title("Price: a very long tail of expensive listings")
sns.histplot(np.log1p(air["price"]), bins=60, ax=axes[1])
axes[1].set_xlabel("log(1 + price)")
axes[1].set_ylabel("Number of listings")
axes[1].set_title("After a log transform")
finish(f"{RES}/airbnb_price_distribution.png")

# Outliers: one box plot per column, each with its own scale
cols = ["price"] + NUM
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, c in zip(axes.ravel(), cols):
    sns.boxplot(y=air[c], ax=ax)
    ax.set_title(c)
    ax.set_ylabel("")
fig.suptitle("Box plots of the numerical columns (dots are outliers)")
finish(f"{RES}/airbnb_boxplots.png")

# Correlations
corr = air[["price"] + NUM].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(9, 7))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation between numerical columns")
finish(f"{RES}/airbnb_correlation.png")

# Price by borough and room type
valid = air[air["price"] > 0]
plt.figure(figsize=(9, 4.8))
sns.boxplot(data=valid, x="neighbourhood_group", y="price", hue="room_type")
plt.yscale("log")
plt.xlabel("Borough")
plt.ylabel("Price per night (USD, log scale)")
plt.title("Price by borough and room type")
finish(f"{RES}/airbnb_price_by_group.png")

# Map of the listings
plt.figure(figsize=(7, 7))
sc = plt.scatter(valid["longitude"], valid["latitude"], c=np.log1p(valid["price"]),
                 cmap="plasma", s=3, alpha=0.6)
plt.colorbar(sc, label="log(1 + price)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Airbnb listings in New York City, colored by price")
finish(f"{RES}/airbnb_price_map.png")

# ==== 3. Cleaning experiment: does each cleaning step help? ====
STEPS_REG = [
    "Raw: numeric columns only, rows with missing values dropped from training",
    "+ missing reviews_per_month filled with 0",
    "+ invalid prices (0) removed from training",
    "+ One-Hot encoding of borough and room type",
    "+ extreme prices (above the 99th percentile) removed from training",
    "+ extreme values of the other columns clipped at the 99th percentile",
    "+ log transform of the price",
]


def build_reg(train, test, step):
    train, test = train.copy(), test.copy()
    p99 = train.loc[train["price"] > 0, "price"].quantile(0.99)
    test = test[(test["price"] > 0) & (test["price"] <= p99)]

    if step >= 1:
        train["reviews_per_month"] = train["reviews_per_month"].fillna(0)
        test["reviews_per_month"] = test["reviews_per_month"].fillna(0)
    else:
        train = train.dropna(subset=NUM)
        test[NUM] = test[NUM].fillna(train[NUM].median())
    if step >= 2:
        train = train[train["price"] > 0]
    if step >= 4:
        train = train[train["price"] <= p99]
    train = train.copy()
    if step >= 5:
        limits = train[CLIP_COLS].quantile(0.99)
        train[CLIP_COLS] = train[CLIP_COLS].clip(upper=limits, axis=1)
        test[CLIP_COLS] = test[CLIP_COLS].clip(upper=limits, axis=1)

    cols = NUM + (CAT if step >= 3 else [])
    X_train, X_test = train[cols], test[cols]
    if step >= 3:
        both = pd.concat([X_train, X_test])
        both = pd.get_dummies(both, columns=CAT, drop_first=True, dtype=float)
        X_train, X_test = both.iloc[:len(X_train)], both.iloc[len(X_train):]

    y_train, y_test = train["price"], test["price"]
    if step >= 6:
        y_train = np.log1p(y_train)
    return X_train, y_train, X_test, y_test, p99


REG_METRICS = ["MAE", "RMSE", "R2"]
reg_runs, base_runs = [], []
for seed in SEEDS:
    train, test = train_test_split(air, test_size=0.2, random_state=seed)
    for step in range(len(STEPS_REG)):
        Xtr, ytr, Xte, yte, p99 = build_reg(train, test, step)
        model = LinearRegression().fit(Xtr, ytr)
        pred = model.predict(Xte)
        if step >= 6:
            pred = np.expm1(pred)
        reg_runs.append({"seed": seed, "step": step, **reg_metrics(yte, pred)})
    ref = train[(train["price"] > 0) & (train["price"] <= p99)]["price"]
    base_runs.append(reg_metrics(yte, np.full(len(yte), ref.mean())))

reg_runs = pd.DataFrame(reg_runs)
reg_summary = summarize(reg_runs, REG_METRICS)
reg_deltas = paired_deltas(reg_runs, REG_METRICS, primary="RMSE", lower_is_better=True)
baseline = pd.DataFrame(base_runs).mean()

log("\n===== Cleaning experiment: Linear Regression on Airbnb prices =====")
log("Metrics are in USD, averaged over 5 different splits (mean and standard deviation).")
for i, name in enumerate(STEPS_REG):
    log(f"S{i}: {name}")
log("\nResults per step:")
log(reg_summary.round(3).to_string())
log("\nChange from the previous step (negative MAE and RMSE and positive R2 mean better):")
log(reg_deltas.round(3).to_string())
log("better_in counts the splits where the RMSE became lower.")
log("\nBaseline that always predicts the average price:")
log(baseline[REG_METRICS].round(3).to_string())
total = reg_summary.iloc[-1][[m + "_mean" for m in REG_METRICS]] - reg_summary.iloc[0][[m + "_mean" for m in REG_METRICS]]
log("\nTotal change from the raw version (S0) to the last version:")
log(total.round(3).to_string())

reg_summary.to_csv(f"{RES}/regression_cleaning_comparison.csv")
reg_deltas.to_csv(f"{RES}/regression_cleaning_changes.csv")
plot_steps(reg_summary, REG_METRICS, "S", f"{RES}/regression_cleaning_comparison.png",
           "Linear Regression: effect of each cleaning step (mean and standard deviation over 5 splits)")

# ==== 4. Final Linear Regression ====
train, test = train_test_split(air, test_size=0.2, random_state=RANDOM_STATE)
Xtr, ytr, Xte, yte, p99 = build_reg(train, test, len(STEPS_REG) - 1)

scaler_reg = StandardScaler()
Xtr_s = scaler_reg.fit_transform(Xtr)
Xte_s = scaler_reg.transform(Xte)

lin_reg = LinearRegression()
lin_reg.fit(Xtr_s, ytr)
pred_log = lin_reg.predict(Xte_s)
pred = np.expm1(pred_log)

final_reg = reg_metrics(yte, pred)
base_pred = np.full(len(yte), np.expm1(ytr).mean())
base_reg = reg_metrics(yte, base_pred)

log("\n===== Final Linear Regression (main split) =====")
log(f"Training listings: {len(Xtr)}, test listings: {len(Xte)}, price limit (99th percentile): {p99:.0f} USD")
log(f"{'Metric':<8}{'Linear Regression':>20}{'Always predict the mean':>26}")
for m in ["MAE", "MSE", "RMSE", "R2"]:
    log(f"{m:<8}{final_reg[m]:>20.4f}{base_reg[m]:>26.4f}")
log(f"R2 on the log scale (what the model actually fits): {r2_score(np.log1p(yte), pred_log):.4f}")

# Predicted vs actual
plt.figure(figsize=(6, 6))
plt.scatter(yte, pred, s=4, alpha=0.3)
plt.plot([0, p99], [0, p99], color="red", label="Perfect prediction")
plt.xlim(0, p99)
plt.ylim(0, p99)
plt.xlabel("Actual price (USD)")
plt.ylabel("Predicted price (USD)")
plt.title("Predicted vs actual prices")
plt.legend()
finish(f"{RES}/regression_predicted_vs_actual.png")

# Residuals
plt.figure(figsize=(7, 4.5))
plt.scatter(pred, yte - pred, s=4, alpha=0.3)
plt.axhline(0, color="red")
plt.xlabel("Predicted price (USD)")
plt.ylabel("Error: actual minus predicted (USD)")
plt.title("Prediction errors")
finish(f"{RES}/regression_residuals.png")

# What drives the price: effect of +1 standard deviation of each feature
coef = pd.Series(lin_reg.coef_, index=Xtr.columns)
effect = ((np.exp(coef) - 1) * 100).sort_values()
plt.figure(figsize=(8, 5.5))
plt.barh(effect.index, effect.values, color=["tab:red" if v < 0 else "tab:green" for v in effect.values])
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Change in price when the feature rises by one standard deviation (%)")
plt.title("What drives Airbnb prices in the model")
finish(f"{RES}/regression_feature_effects.png")

# ==== Part B: Classification with Logistic Regression (EmpatheticDialogues) ====
emp = pd.read_csv(f"{DATA}/empathetic.csv")
log("\n\n===== EMPATHETIC DIALOGUES DATASET =====")
log(f"Shape: {emp.shape}")
print("\nColumn data types:")
print(emp.dtypes)
print(emp.head())
print("\nMissing values per column:")
print(emp.isnull().sum())
emp = emp.dropna(subset=["situation", "emotion"])
emp_all = emp.copy()

# How the 32 emotions were grouped
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
finish(f"{RES}/empathetic_emotion_groups.png")

# Binary target: 1 = positive emotion, 0 = negative emotion
emp = emp[emp["emotion"].isin(POSITIVE + NEGATIVE)].reset_index(drop=True)
emp["label"] = emp["emotion"].isin(POSITIVE).astype(int)
log("Classification target: label (1 = positive, 0 = negative)")
log(emp["label"].value_counts().to_string())
log(f"Duplicated situation texts: {emp['situation'].duplicated().sum()}")

plt.figure(figsize=(7, 4))
sns.histplot(emp["situation"].str.split().str.len(), bins=40)
plt.xlabel("Words per situation")
plt.ylabel("Number of conversations")
plt.title("Length of the situation texts")
finish(f"{RES}/empathetic_text_length.png")

# ==== 2. Text cleaning experiment: does each cleaning step help? ====
STEPS_TXT = [
    "Raw text: split on spaces, case and punctuation kept",
    "+ lowercase",
    "+ remove links, numbers and punctuation, expand n't to not",
    "+ remove stop words (negations like not and never are kept)",
    "+ simple stemming (cut common word endings)",
    "+ drop duplicate and very short texts from the training set",
]
NEGATIONS = {"no", "not", "nor", "never", "none", "nothing", "nobody", "neither", "cannot", "without"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATIONS


def clean_symbols(text):
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("_comma_", " ")
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def remove_stop_words(text):
    return " ".join(w for w in text.split() if w not in STOP_WORDS)


def simple_stem(w):
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 4 and w.endswith("ly"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def apply_text_steps(series, step):
    s = series
    if step >= 1:
        s = s.str.lower()
    if step >= 2:
        s = s.apply(clean_symbols)
    if step >= 3:
        s = s.apply(remove_stop_words)
    if step >= 4:
        s = s.apply(lambda t: " ".join(simple_stem(w) for w in t.split()))
    return s


texts = {s: apply_text_steps(emp["situation"], s) for s in range(5)}


def text_data(step, seed):
    tr_idx, te_idx = train_test_split(emp.index, test_size=0.2, random_state=seed,
                                      stratify=emp["label"])
    col = texts[min(step, 4)]
    tr_text, te_text = col.loc[tr_idx], col.loc[te_idx]
    y_tr, y_te = emp.loc[tr_idx, "label"], emp.loc[te_idx, "label"]
    if step == 5:
        keep = ~tr_text.duplicated() & (tr_text.str.split().str.len() >= 3)
        tr_text, y_tr = tr_text[keep], y_tr[keep]
    vec = TfidfVectorizer(max_features=5000, lowercase=False, token_pattern=r"\S+")
    return vec.fit_transform(tr_text), y_tr, vec.transform(te_text), y_te, vec


CLF_METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
txt_runs = []
for seed in SEEDS:
    for step in range(len(STEPS_TXT)):
        Xtr, ytr, Xte, yte, vec = text_data(step, seed)
        model = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        txt_runs.append({"seed": seed, "step": step,
                         **clf_metrics(yte, model.predict(Xte), model.decision_function(Xte))})

txt_runs = pd.DataFrame(txt_runs)
txt_summary = summarize(txt_runs, CLF_METRICS)
txt_deltas = paired_deltas(txt_runs, CLF_METRICS, primary="F1", lower_is_better=False)

log("\n===== Cleaning experiment: Logistic Regression on the emotion texts =====")
log("Averaged over 5 different splits (mean and standard deviation).")
for i, name in enumerate(STEPS_TXT):
    log(f"T{i}: {name}")
log("\nResults per step:")
log(txt_summary.round(4).to_string())
log("\nChange from the previous step:")
log(txt_deltas.round(4).to_string())
log("better_in counts the splits where the F1 became higher.")
first, last = txt_summary.iloc[0], txt_summary.iloc[-1]
log("\nTotal change from the raw text (T0) to the last version (T5):")
log(pd.Series({m: last[m + "_mean"] - first[m + "_mean"] for m in CLF_METRICS}).round(4).to_string())

txt_summary.to_csv(f"{RES}/text_cleaning_comparison.csv")
txt_deltas.to_csv(f"{RES}/text_cleaning_changes.csv")
plot_steps(txt_summary, ["Accuracy", "F1", "ROC_AUC"], "T", f"{RES}/text_cleaning_comparison.png",
           "Logistic Regression: effect of each text-cleaning step (mean and standard deviation over 5 splits)")

# ==== 3. Choosing the model and the final evaluation ====
best_step = int(txt_summary["F1_mean"].idxmax())
log(f"\nText preparation chosen by the experiment (best average F1): T{best_step}: {STEPS_TXT[best_step]}")

Xtr, ytr, Xte, yte, vec = text_data(best_step, RANDOM_STATE)
log_reg = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
tree = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(Xtr, ytr)

scores = {"Logistic Regression": log_reg.decision_function(Xte),
          "Decision Tree": tree.predict_proba(Xte)[:, 1]}
preds = {"Logistic Regression": log_reg.predict(Xte), "Decision Tree": tree.predict(Xte)}

majority = max(yte.mean(), 1 - yte.mean())
log("\n===== Model comparison on the main split =====")
rows = []
for name in scores:
    rows.append({"Model": name, **clf_metrics(yte, preds[name], scores[name])})
comparison = pd.DataFrame(rows).set_index("Model")
log(comparison.round(4).to_string())
log(f"Baseline accuracy if the model always answers with the majority class: {majority:.4f}")
comparison.to_csv(f"{RES}/classification_model_comparison.csv")

y_pred = preds["Logistic Regression"]
cm = confusion_matrix(yte, y_pred)
tn, fp, fn, tp = cm.ravel()
log("\n===== Final Logistic Regression =====")
log(f"Accuracy:  {accuracy_score(yte, y_pred):.4f}")
log(f"Confusion Matrix:\n{cm}")
log(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
log("\nClassification report:")
log(classification_report(yte, y_pred, target_names=["Negative", "Positive"], digits=4))

# Confusion matrix with names and percentages
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
annot = np.array([[f"{cm[i, j]}\n{cm_pct[i, j]:.1f}%" for j in range(2)] for i in range(2)])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
            xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted emotion")
plt.ylabel("Actual emotion")
plt.title("Confusion Matrix")
finish(f"{RES}/confusion_matrix.png")

# ROC curves
plt.figure(figsize=(5.5, 5))
for name in scores:
    fpr, tpr, _ = roc_curve(yte, scores[name])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(yte, scores[name]):.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guessing")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curves")
plt.legend()
finish(f"{RES}/roc_curves.png")

# Words that push the prediction toward negative (red) or positive (green)
words = vec.get_feature_names_out()
coefs = log_reg.coef_[0]
order = np.argsort(coefs)
idx = np.concatenate([order[:10], order[-10:]])
plt.figure(figsize=(7, 7))
plt.barh(words[idx], coefs[idx], color=["tab:red"] * 10 + ["tab:green"] * 10)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Weight in the model")
plt.title("Words the model uses to tell negative from positive")
finish(f"{RES}/classification_top_words.png")

# ==== Part C: Clustering with K-Means (NYC Airbnb, without the price) ====
log("\n\n===== K-Means Clustering =====")
cl = air[air["price"] > 0].copy()
cl["reviews_per_month"] = cl["reviews_per_month"].fillna(0)
log(f"Listings used: {len(cl)}")

CL_NUM = ["latitude", "longitude", "minimum_nights", "number_of_reviews",
          "reviews_per_month", "calculated_host_listings_count", "availability_365"]
cl_raw = pd.concat([cl[CL_NUM], pd.get_dummies(cl["room_type"], dtype=float)], axis=1)
cl_clipped = cl_raw.copy()
limits = cl_raw[CLIP_COLS].quantile(0.99)
cl_clipped[CLIP_COLS] = cl_clipped[CLIP_COLS].clip(upper=limits, axis=1)
X_final = StandardScaler().fit_transform(cl_clipped)

# ==== 2. Choosing K with the Elbow Method ====
k_values = list(range(1, 11))
inertias, sils = [], []
for k in k_values:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_final)
    inertias.append(km.inertia_)
    if k > 1:
        sils.append(silhouette_score(X_final, km.labels_, sample_size=5000, random_state=RANDOM_STATE))
    else:
        sils.append(np.nan)

# Elbow: the point farthest from the line joining the first and the last points
p1 = np.array([k_values[0], inertias[0]])
p2 = np.array([k_values[-1], inertias[-1]])
distances = []
for k, inertia in zip(k_values, inertias):
    p = np.array([k, inertia])
    d = p2 - p1
    v = p1 - p
    distances.append(abs(d[0] * v[1] - d[1] * v[0]) / np.linalg.norm(d))
optimal_k = k_values[int(np.argmax(distances))]

k_table = pd.DataFrame({"K": k_values, "Inertia": inertias, "Silhouette": sils}).set_index("K")
log("Inertia and Silhouette Score for each K:")
log(k_table.round(4).to_string())
log(f"Optimal K (Elbow): {optimal_k}")
k_table.to_csv(f"{RES}/k_selection.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(k_values, inertias, marker="o")
axes[0].axvline(optimal_k, color="red", linestyle="--", label=f"Chosen K = {optimal_k}")
axes[0].set_xlabel("Number of clusters (K)")
axes[0].set_ylabel("Inertia (lower = tighter clusters)")
axes[0].set_title("Elbow Method")
axes[0].legend()
axes[1].plot(k_values[1:], sils[1:], marker="o", color="tab:green")
axes[1].axvline(optimal_k, color="red", linestyle="--")
axes[1].set_xlabel("Number of clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score for each K")
finish(f"{RES}/elbow_plot.png")

# ==== 3. Does preparing the features help? ====
versions = {
    "C0 Raw features (not scaled, not clipped)": cl_raw.values,
    "C1 + scaled (StandardScaler)": StandardScaler().fit_transform(cl_raw),
    "C2 + extreme values clipped at the 99th percentile": X_final,
}
rows = []
for name, X in versions.items():
    km = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10).fit(X)
    rows.append({
        "Version": name,
        "Silhouette": silhouette_score(X, km.labels_, sample_size=10000, random_state=RANDOM_STATE),
        "Davies-Bouldin": davies_bouldin_score(X, km.labels_),
        "Calinski-Harabasz": calinski_harabasz_score(X, km.labels_),
    })
cluster_prep = pd.DataFrame(rows).set_index("Version")
log("\nClustering quality for each version of the features (K = %d):" % optimal_k)
log("Higher is better for Silhouette and Calinski-Harabasz, lower is better for Davies-Bouldin.")
log(cluster_prep.round(4).to_string())
var_share = cl_raw.var() / cl_raw.var().sum()
log(f"Warning: in the raw version the column {var_share.idxmax()} alone makes up {var_share.max() * 100:.1f}% of the total variance, "
    "so K-Means mostly clusters on that single column. Its scores can look better only because the groups are easy to separate on one column, "
    "not because the clusters are more meaningful.")
cluster_prep.to_csv(f"{RES}/clustering_preparation_comparison.csv")

# ==== 4. Final clusters ====
kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10).fit(X_final)
labels = kmeans.labels_
log("\n===== Final K-Means =====")
log(f"Silhouette Score (sample of 10,000 listings): {silhouette_score(X_final, labels, sample_size=10000, random_state=RANDOM_STATE):.4f}")
log(f"Davies-Bouldin Index: {davies_bouldin_score(X_final, labels):.4f}")
log(f"Calinski-Harabasz Index: {calinski_harabasz_score(X_final, labels):.2f}")

cl["cluster"] = labels
grouped = cl.groupby("cluster")
cluster_summary = pd.DataFrame({
    "listings": grouped.size(),
    "median_price": grouped["price"].median(),
    "median_min_nights": grouped["minimum_nights"].median(),
    "median_reviews": grouped["number_of_reviews"].median(),
    "median_availability": grouped["availability_365"].median(),
    "median_host_listings": grouped["calculated_host_listings_count"].median(),
    "main_room_type": grouped["room_type"].agg(lambda s: s.value_counts().index[0]),
    "main_borough": grouped["neighbourhood_group"].agg(lambda s: s.value_counts().index[0]),
})
log("\nCluster summary:")
log(cluster_summary.to_string())
cluster_summary.to_csv(f"{RES}/cluster_summary.csv")

# Clusters on the map
plt.figure(figsize=(7, 7))
for c in range(optimal_k):
    members = cl["cluster"] == c
    plt.scatter(cl.loc[members, "longitude"], cl.loc[members, "latitude"], s=3, alpha=0.6, label=f"Cluster {c}")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"K-Means clusters on the map (K = {optimal_k})")
plt.legend(markerscale=4)
finish(f"{RES}/clusters.png")

# Price inside each cluster (the price was not used to build the clusters)
plt.figure(figsize=(7, 4.5))
sns.boxplot(data=cl, x="cluster", y="price")
plt.yscale("log")
plt.xlabel("Cluster")
plt.ylabel("Price per night (USD, log scale)")
plt.title("Price inside each cluster")
finish(f"{RES}/cluster_prices.png")

# ==== Save all numbers ====
with open(f"{RES}/metrics.txt", "w") as f:
    f.write("\n".join(log_lines))
print("\nAll numbers were saved to results/metrics.txt")
