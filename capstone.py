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
SEEDS = [42, 43, 44, 45, 46]      # every before/after experiment is repeated on 5 different splits
MIN_WINS = 4                      # a cleaning step is kept only if it helps in at least 4 of the 5 splits
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
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close()


def titled(ax, main, hint=None):
    """A title plus a small gray line that says how to read the plot."""
    ax.set_title(main, fontsize=11, pad=24 if hint else 6)
    if hint:
        ax.text(0.5, 1.03, hint, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8, color="dimgray")


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


def make_cached(func):
    """Remember the result for each (cleaning steps, split), so nothing is trained twice."""
    cache = {}

    def wrapper(ops, seed):
        key = (frozenset(ops), seed)
        if key not in cache:
            cache[key] = func(frozenset(ops), seed)
        return cache[key]
    return wrapper


def cumulative_runs(evaluate, ops_list):
    """Version 0 is the raw data; version i adds the first i cleaning steps."""
    rows = []
    for i in range(len(ops_list) + 1):
        for seed in SEEDS:
            rows.append({"seed": seed, "step": i, **evaluate(ops_list[:i], seed)})
    return pd.DataFrame(rows)


def summarize(runs, metrics):
    """Mean and standard deviation of each metric over the splits, for every version."""
    table = runs.groupby("step")[metrics].agg(["mean", "std"])
    table.columns = [f"{m}_{s}" for m, s in table.columns]
    return table


def paired_deltas(runs, metrics, primary, lower_is_better):
    """Change from the previous version, measured on the same split, so the comparison is fair."""
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


def greedy_select(evaluate, ops, primary, lower_is_better):
    """Go through the cleaning steps in order and keep a step only if it clearly helps."""
    chosen, rows = [], []
    for op in ops:
        candidate = chosen + [op]
        diffs = [evaluate(candidate, s)[primary] - evaluate(chosen, s)[primary] for s in SEEDS]
        wins = sum(d < 0 for d in diffs) if lower_is_better else sum(d > 0 for d in diffs)
        keep = wins >= MIN_WINS
        rows.append({"step": op, f"{primary}_change": np.mean(diffs),
                     "better_in": f"{wins} of {len(SEEDS)} splits",
                     "decision": "kept" if keep else "dropped"})
        if keep:
            chosen = candidate
    return chosen, pd.DataFrame(rows).set_index("step")


def mean_std(evaluate, ops, metric):
    values = [evaluate(ops, s)[metric] for s in SEEDS]
    return float(np.mean(values)), float(np.std(values, ddof=1))


def plot_steps(summary, chosen, metrics, labels, hints, filename, title):
    """Metrics after each added cleaning step, plus the final chosen version (red star)."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.4 * len(metrics), 5))
    x = list(range(len(summary)))
    for ax, m in zip(np.atleast_1d(axes), metrics):
        ax.errorbar(x, summary[m + "_mean"], yerr=summary[m + "_std"], marker="o", capsize=3,
                    label="Steps added one by one")
        ax.errorbar([len(summary)], [chosen[m + "_mean"]], yerr=[chosen[m + "_std"]], marker="*",
                    markersize=14, color="tab:red", capsize=3, label="Chosen by the rule")
        ax.set_xticks(x + [len(summary)])
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        titled(ax, m, hints[m])
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle(title, y=1.02)
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

# ==== 2. Exploratory data analysis: what is wrong with the data? ====
# 1) Missing values
miss = air.isnull().sum().sort_values(ascending=False)
miss = miss[miss > 0]
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.barh(miss.index[::-1], miss.values[::-1])
ax.set_xlabel("Number of missing values")
titled(ax, "Where the data is missing",
       "reviews_per_month and last_review are empty for listings that never got a review")
finish(f"{RES}/airbnb_missing_values.png")

# 2) Price before and after cleaning
p99_all = air.loc[air["price"] > 0, "price"].quantile(0.99)
bad_price = (air["price"] == 0) | (air["price"] > p99_all)
price_clean = air.loc[~bad_price, "price"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
sns.histplot(air["price"], bins=60, ax=axes[0])
axes[0].set_yscale("log")
axes[0].set_xlabel("Price per night (USD)")
axes[0].set_ylabel("Number of listings (log scale)")
titled(axes[0], "BEFORE: all prices",
       f"a few listings cost up to {air['price'].max():,} USD, and {(air['price'] == 0).sum()} cost 0")
sns.histplot(price_clean, bins=60, ax=axes[1])
axes[1].set_xlabel("Price per night (USD)")
axes[1].set_ylabel("Number of listings")
titled(axes[1], "AFTER: price above 0 and up to the 99th percentile",
       f"{bad_price.sum()} listings ({bad_price.mean() * 100:.1f}%) removed, limit = {p99_all:.0f} USD")
sns.histplot(np.log1p(price_clean), bins=60, ax=axes[2])
axes[2].set_xlabel("log(1 + price)")
axes[2].set_ylabel("Number of listings")
titled(axes[2], "After a log transform (one of the cleaning steps we test)",
       "the long tail becomes a bell shape, which suits a linear model better")
finish(f"{RES}/airbnb_price_before_after.png")

# 3) Outliers before and after (illustration on the full data; the model uses limits from the training set only)
cols = ["price"] + CLIP_COLS
fig, axes = plt.subplots(2, len(cols), figsize=(3.2 * len(cols), 7))
for j, c in enumerate(cols):
    before = air[c]
    if c == "price":
        after = price_clean
    else:
        after = air[c].clip(upper=air[c].quantile(0.99))
    sns.boxplot(y=before, ax=axes[0, j], color="tab:red")
    sns.boxplot(y=after, ax=axes[1, j], color="tab:green")
    axes[0, j].set_title(c, fontsize=10)
    axes[0, j].set_ylabel("BEFORE" if j == 0 else "")
    axes[1, j].set_ylabel("AFTER" if j == 0 else "")
fig.suptitle("Outliers before (red) and after (green) the cleaning rules\n"
             "dots far above the box are outliers; after cleaning the extreme values are cut at the 99th percentile",
             y=1.0)
finish(f"{RES}/airbnb_outliers_before_after.png")

# 4) What explains the price
corr = air[["price"] + NUM].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(9, 7.2))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
titled(ax, "Correlation between the numerical columns",
       "red = move together, blue = move in opposite directions, near 0 = unrelated")
finish(f"{RES}/airbnb_correlation.png")

valid = air[air["price"] > 0]
fig, ax = plt.subplots(figsize=(9.5, 5.2))
sns.boxplot(data=valid, x="neighbourhood_group", y="price", hue="room_type", ax=ax)
ax.set_yscale("log")
ax.set_xlabel("Borough")
ax.set_ylabel("Price per night (USD, log scale)")
titled(ax, "Price by borough and room type",
       "entire homes cost more than private rooms in every borough, and Manhattan costs the most")
finish(f"{RES}/airbnb_price_by_group.png")

fig, ax = plt.subplots(figsize=(7.5, 7.5))
sc = ax.scatter(valid["longitude"], valid["latitude"], c=np.log1p(valid["price"]),
                cmap="plasma", s=3, alpha=0.6)
plt.colorbar(sc, ax=ax, label="log(1 + price)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
titled(ax, "Airbnb listings in New York City, colored by price", "brighter = more expensive")
finish(f"{RES}/airbnb_price_map.png")

# 5) Table: how many rows the outlier rules touch
rows = [{"column": "price", "max_before": air["price"].max(), "limit": round(p99_all, 1),
         "rows_affected": int(bad_price.sum()),
         "rule": "listings with price 0 or above the limit are removed from training"}]
for c in CLIP_COLS:
    limit = air[c].quantile(0.99)
    rows.append({"column": c, "max_before": air[c].max(), "limit": round(limit, 1),
                 "rows_affected": int((air[c] > limit).sum()),
                 "rule": "values above the limit are cut to the limit"})
outlier_table = pd.DataFrame(rows).set_index("column")
outlier_table["share_of_rows_%"] = (outlier_table["rows_affected"] / len(air) * 100).round(2)
log("\nOutlier rules (limits are the 99th percentile; the models compute them on the training set only):")
log(outlier_table.to_string())
outlier_table.to_csv(f"{RES}/outlier_summary.csv")

# ==== 3. Cleaning experiment: does each cleaning step help? ====
OPS_REG = ["fill_reviews", "drop_zero_price", "drop_extreme_price", "clip_features", "log_price"]
OP_TEXT_REG = {
    "fill_reviews": "missing reviews_per_month filled with 0 (instead of dropping those rows)",
    "drop_zero_price": "listings with price 0 removed from training",
    "drop_extreme_price": "prices above the 99th percentile removed from training",
    "clip_features": "extreme values of the other columns cut at the 99th percentile",
    "log_price": "model trained on log(1 + price)",
}
LABELS_REG = ["Raw", "+ fill missing\nreviews", "+ drop\nprice 0", "+ drop extreme\nprices",
              "+ clip other\ncolumns", "+ log of price", "Chosen"]
SPLITS_REG = {s: train_test_split(air, test_size=0.2, random_state=s) for s in SEEDS}


def build_reg(train, test, ops, use_cat=True):
    ops = set(ops)
    train, test = train.copy(), test.copy()
    p99 = train.loc[train["price"] > 0, "price"].quantile(0.99)
    test = test[(test["price"] > 0) & (test["price"] <= p99)]

    if "fill_reviews" in ops:
        train["reviews_per_month"] = train["reviews_per_month"].fillna(0)
        test["reviews_per_month"] = test["reviews_per_month"].fillna(0)
    else:
        train = train.dropna(subset=NUM)
        test[NUM] = test[NUM].fillna(train[NUM].median())
    if "drop_zero_price" in ops:
        train = train[train["price"] > 0]
    if "drop_extreme_price" in ops:
        train = train[train["price"] <= p99]
    train = train.copy()
    if "clip_features" in ops:
        limits = train[CLIP_COLS].quantile(0.99)
        train[CLIP_COLS] = train[CLIP_COLS].clip(upper=limits, axis=1)
        test[CLIP_COLS] = test[CLIP_COLS].clip(upper=limits, axis=1)

    cols = NUM + (CAT if use_cat else [])
    X_train, X_test = train[cols], test[cols]
    if use_cat:
        both = pd.concat([X_train, X_test])
        both = pd.get_dummies(both, columns=CAT, drop_first=True, dtype=float)
        X_train, X_test = both.iloc[:len(X_train)], both.iloc[len(X_train):]

    y_train, y_test = train["price"], test["price"]
    if "log_price" in ops:
        y_train = np.log1p(y_train)
    return X_train, y_train, X_test, y_test, p99


def eval_reg(ops, seed, use_cat=True):
    train, test = SPLITS_REG[seed]
    Xtr, ytr, Xte, yte, p99 = build_reg(train, test, ops, use_cat)
    pred = LinearRegression().fit(Xtr, ytr).predict(Xte)
    if "log_price" in ops:
        pred = np.expm1(pred)
    return reg_metrics(yte, pred)


reg_eval = make_cached(eval_reg)
REG_METRICS = ["MAE", "RMSE", "R2"]

log("\n===== Cleaning experiment: Linear Regression on Airbnb prices =====")
log("Metrics are in USD, averaged over 5 different splits. All versions use the same columns.")
for i, op in enumerate(OPS_REG, 1):
    log(f"Step {i}, {op}: {OP_TEXT_REG[op]}")

reg_runs = cumulative_runs(reg_eval, OPS_REG)
reg_summary = summarize(reg_runs, REG_METRICS)
reg_deltas = paired_deltas(reg_runs, REG_METRICS, primary="RMSE", lower_is_better=True)
log("\nResults for the raw data and after each added step:")
log(reg_summary.round(3).to_string())
log("\nChange from the previous version (a lower MAE and RMSE and a higher R2 are better):")
log(reg_deltas.round(3).to_string())
log("better_in counts the splits where the RMSE became lower.")

chosen_reg, reg_choice = greedy_select(reg_eval, OPS_REG, "RMSE", True)
log("\nSelection rule: keep a step only if it lowers the RMSE in at least 4 of the 5 splits.")
log(reg_choice.round(3).to_string())
log(f"Chosen cleaning steps: {chosen_reg if chosen_reg else 'none'}")

chosen_rows = pd.DataFrame([reg_eval(chosen_reg, s) for s in SEEDS])
chosen_reg_summary = pd.Series({f"{m}_{k}": getattr(chosen_rows[m], k)() for m in REG_METRICS for k in ("mean", "std")})
raw_reg_summary = reg_summary.iloc[0]
log("\nRaw data against the chosen cleaning:")
comparison_reg = pd.DataFrame({
    "raw": [raw_reg_summary[m + "_mean"] for m in REG_METRICS],
    "chosen": [chosen_reg_summary[m + "_mean"] for m in REG_METRICS]}, index=REG_METRICS)
comparison_reg["change"] = comparison_reg["chosen"] - comparison_reg["raw"]
comparison_reg["change_%"] = comparison_reg["change"] / comparison_reg["raw"].abs() * 100
log(comparison_reg.round(3).to_string())

# Adding the borough and room type columns is a feature choice, so it is measured on the raw data
feat_rows = []
for use_cat in (False, True):
    vals = pd.DataFrame([eval_reg(frozenset(), s, use_cat) for s in SEEDS])[REG_METRICS].mean()
    feat_rows.append({"columns": "numeric + borough + room type" if use_cat else "numeric columns only", **vals})
feat_table = pd.DataFrame(feat_rows).set_index("columns")
log("\nEffect of the columns used (a feature choice, not cleaning), on the raw data:")
log(feat_table.round(3).to_string())

reg_summary.to_csv(f"{RES}/regression_cleaning_comparison.csv")
reg_deltas.to_csv(f"{RES}/regression_cleaning_changes.csv")
reg_choice.to_csv(f"{RES}/regression_cleaning_choice.csv")
feat_table.to_csv(f"{RES}/regression_feature_choice.csv")
plot_steps(reg_summary, chosen_reg_summary, REG_METRICS, LABELS_REG,
           {"MAE": "average error in USD, lower is better", "RMSE": "typical error in USD, lower is better",
            "R2": "share of price variation explained, higher is better"},
           f"{RES}/regression_cleaning_comparison.png",
           "Linear Regression: what each cleaning step is worth (mean and standard deviation over 5 splits)")

# ==== 4. Final Linear Regression ====
train, test = train_test_split(air, test_size=0.2, random_state=RANDOM_STATE)
Xtr, ytr, Xte, yte, p99 = build_reg(train, test, chosen_reg)

scaler_reg = StandardScaler()
Xtr_s = scaler_reg.fit_transform(Xtr)
Xte_s = scaler_reg.transform(Xte)

lin_reg = LinearRegression()
lin_reg.fit(Xtr_s, ytr)
pred = lin_reg.predict(Xte_s)
if "log_price" in chosen_reg:
    pred = np.expm1(pred)

final_reg = reg_metrics(yte, pred)
train_prices = np.expm1(ytr) if "log_price" in chosen_reg else ytr
base_pred = np.full(len(yte), train_prices.mean())
base_reg = reg_metrics(yte, base_pred)

log("\n===== Final Linear Regression (main split) =====")
log(f"Training listings: {len(Xtr)}, test listings: {len(Xte)}, price limit (99th percentile): {p99:.0f} USD")
log(f"{'Metric':<8}{'Linear Regression':>20}{'Always predict the mean':>26}")
for m in ["MAE", "MSE", "RMSE", "R2"]:
    log(f"{m:<8}{final_reg[m]:>20.4f}{base_reg[m]:>26.4f}")

# Baseline (before) next to the model (after)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), sharey=True)
for ax, p, name, m in [(axes[0], base_pred, "BEFORE: no model, always predict the average", base_reg),
                       (axes[1], pred, "AFTER: Linear Regression", final_reg)]:
    ax.scatter(yte, p, s=4, alpha=0.3)
    ax.plot([0, p99], [0, p99], color="red", label="Perfect prediction")
    ax.set_xlim(0, p99)
    ax.set_ylim(0, p99)
    ax.set_xlabel("Actual price (USD)")
    titled(ax, name, f"R2 = {m['R2']:.2f}, typical error (RMSE) = {m['RMSE']:.0f} USD")
axes[0].set_ylabel("Predicted price (USD)")
axes[1].legend(loc="upper left")
fig.suptitle("Predicted against actual prices: the closer the dots are to the red line, the better", y=1.02)
finish(f"{RES}/regression_predicted_vs_actual.png")

# Residuals
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.scatter(pred, yte - pred, s=4, alpha=0.3)
ax.axhline(0, color="red")
ax.set_xlabel("Predicted price (USD)")
ax.set_ylabel("Error: actual minus predicted (USD)")
titled(ax, "Prediction errors", "errors should be spread around 0; a pattern means the model misses something")
finish(f"{RES}/regression_residuals.png")

# What drives the price
coef = pd.Series(lin_reg.coef_, index=Xtr.columns)
if "log_price" in chosen_reg:
    effect = ((np.exp(coef) - 1) * 100).sort_values()
    xlabel = "Change in price when the feature rises by one standard deviation (%)"
else:
    effect = coef.sort_values()
    xlabel = "Change in price when the feature rises by one standard deviation (USD)"
fig, ax = plt.subplots(figsize=(8.5, 5.8))
ax.barh(effect.index, effect.values, color=["tab:red" if v < 0 else "tab:green" for v in effect.values])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel(xlabel)
titled(ax, "What drives Airbnb prices in the model", "green raises the price, red lowers it")
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

counts = emp_all["emotion"].value_counts()
bar_colors = ["tab:green" if e in POSITIVE else "tab:red" if e in NEGATIVE else "tab:gray"
              for e in counts.index]
fig, ax = plt.subplots(figsize=(8.5, 9.5))
ax.barh(counts.index[::-1], counts.values[::-1], color=bar_colors[::-1])
ax.set_xlabel("Number of conversations")
ax.legend(handles=[Patch(color="tab:green", label="Positive"),
                   Patch(color="tab:red", label="Negative"),
                   Patch(color="tab:gray", label="Left out (ambiguous)")])
titled(ax, "The 32 emotions and how they were grouped",
       "this is how the binary target (positive or negative) was built")
finish(f"{RES}/empathetic_emotion_groups.png")

emp = emp[emp["emotion"].isin(POSITIVE + NEGATIVE)].reset_index(drop=True)
n_before = len(emp)
emp = emp.drop_duplicates(subset="situation").reset_index(drop=True)
emp["label"] = emp["emotion"].isin(POSITIVE).astype(int)
log(f"Duplicated situation texts removed before splitting: {n_before - len(emp)}; {len(emp)} conversations remain")
log("Classification target: label (1 = positive, 0 = negative)")
log(emp["label"].value_counts().to_string())

fig, ax = plt.subplots(figsize=(7.5, 4.5))
sns.histplot(emp["situation"].str.split().str.len(), bins=40, ax=ax)
ax.set_xlabel("Words per situation")
ax.set_ylabel("Number of conversations")
titled(ax, "Length of the situation texts", "most texts are short, so each word carries a lot of weight")
finish(f"{RES}/empathetic_text_length.png")

# ==== 2. Text cleaning experiment: does each cleaning step help? ====
OPS_TXT = ["lowercase", "clean_symbols", "stop_words", "stem"]
OP_TEXT_TXT = {
    "lowercase": "all letters made lowercase, so Happy and happy are one word",
    "clean_symbols": "links, numbers and punctuation removed, n't expanded to not",
    "stop_words": "common words removed (negations like not and never are kept)",
    "stem": "simple stemming: common word endings are cut",
}
LABELS_TXT = ["Raw text", "+ lowercase", "+ clean\nsymbols", "+ remove\nstop words", "+ stemming", "Chosen"]
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


_text_cache = {}


def texts_for(ops):
    key = frozenset(ops)
    if key not in _text_cache:
        s = emp["situation"]
        if "lowercase" in key:
            s = s.str.lower()
        if "clean_symbols" in key:
            s = s.apply(clean_symbols)
        if "stop_words" in key:
            s = s.apply(remove_stop_words)
        if "stem" in key:
            s = s.apply(lambda t: " ".join(simple_stem(w) for w in t.split()))
        _text_cache[key] = s
    return _text_cache[key]


SPLITS_TXT = {s: train_test_split(emp.index, test_size=0.2, random_state=s, stratify=emp["label"])
              for s in SEEDS}


def text_data(ops, seed):
    tr_idx, te_idx = SPLITS_TXT[seed]
    col = texts_for(ops)
    vec = TfidfVectorizer(max_features=5000, lowercase=False, token_pattern=r"\S+")
    Xtr = vec.fit_transform(col.loc[tr_idx])
    return Xtr, emp.loc[tr_idx, "label"], vec.transform(col.loc[te_idx]), emp.loc[te_idx, "label"], vec


def eval_txt(ops, seed):
    Xtr, ytr, Xte, yte, vec = text_data(ops, seed)
    model = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    return clf_metrics(yte, model.predict(Xte), model.decision_function(Xte))


txt_eval = make_cached(eval_txt)
CLF_METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]

log("\n===== Cleaning experiment: Logistic Regression on the emotion texts =====")
log("Averaged over 5 different splits.")
for i, op in enumerate(OPS_TXT, 1):
    log(f"Step {i}, {op}: {OP_TEXT_TXT[op]}")

txt_runs = cumulative_runs(txt_eval, OPS_TXT)
txt_summary = summarize(txt_runs, CLF_METRICS)
txt_deltas = paired_deltas(txt_runs, CLF_METRICS, primary="F1", lower_is_better=False)
log("\nResults for the raw text and after each added step:")
log(txt_summary.round(4).to_string())
log("\nChange from the previous version:")
log(txt_deltas.round(4).to_string())
log("better_in counts the splits where the F1 became higher.")

chosen_txt, txt_choice = greedy_select(txt_eval, OPS_TXT, "F1", False)
log("\nSelection rule: keep a step only if it raises the F1 in at least 4 of the 5 splits.")
log(txt_choice.round(4).to_string())
log(f"Chosen text-cleaning steps: {chosen_txt if chosen_txt else 'none'}")

chosen_rows = pd.DataFrame([txt_eval(chosen_txt, s) for s in SEEDS])
chosen_txt_summary = pd.Series({f"{m}_{k}": getattr(chosen_rows[m], k)() for m in CLF_METRICS for k in ("mean", "std")})
raw_txt_summary = txt_summary.iloc[0]
comparison_txt = pd.DataFrame({
    "raw": [raw_txt_summary[m + "_mean"] for m in CLF_METRICS],
    "chosen": [chosen_txt_summary[m + "_mean"] for m in CLF_METRICS]}, index=CLF_METRICS)
comparison_txt["change"] = comparison_txt["chosen"] - comparison_txt["raw"]
log("\nRaw text against the chosen cleaning:")
log(comparison_txt.round(4).to_string())

words_before = len(set(" ".join(emp["situation"]).split()))
words_after = len(set(" ".join(texts_for(chosen_txt)).split()))
log(f"\nDifferent words in the texts: {words_before} before cleaning, {words_after} after")
sample_idx = emp.sample(5, random_state=RANDOM_STATE).index
examples = pd.DataFrame({"before": emp.loc[sample_idx, "situation"], "after": texts_for(chosen_txt).loc[sample_idx]})
log("\nFive texts before and after the chosen cleaning:")
for _, r in examples.iterrows():
    log(f"  BEFORE: {r['before']}")
    log(f"  AFTER:  {r['after']}")
examples.to_csv(f"{RES}/text_cleaning_examples.csv", index=False)

txt_summary.to_csv(f"{RES}/text_cleaning_comparison.csv")
txt_deltas.to_csv(f"{RES}/text_cleaning_changes.csv")
txt_choice.to_csv(f"{RES}/text_cleaning_choice.csv")
plot_steps(txt_summary, chosen_txt_summary, ["Accuracy", "F1", "ROC_AUC"], LABELS_TXT,
           {"Accuracy": "share of correct answers, higher is better", "F1": "balance of precision and recall, higher is better",
            "ROC_AUC": "how well positive and negative are ranked, higher is better"},
           f"{RES}/text_cleaning_comparison.png",
           "Logistic Regression: what each text-cleaning step is worth (mean and standard deviation over 5 splits)")

# ==== 3. Choosing the model and the final evaluation ====
Xtr, ytr, Xte, yte, vec = text_data(chosen_txt, RANDOM_STATE)
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

cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
annot = np.array([[f"{cm[i, j]}\n{cm_pct[i, j]:.1f}%" for j in range(2)] for i in range(2)])
fig, ax = plt.subplots(figsize=(5.8, 4.8))
sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", ax=ax,
            xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
ax.set_xlabel("Predicted emotion")
ax.set_ylabel("Actual emotion")
titled(ax, "Confusion Matrix", "the diagonal is correct answers; the percentage is the share of each actual class")
finish(f"{RES}/confusion_matrix.png")

fig, ax = plt.subplots(figsize=(6, 5.4))
for name in scores:
    fpr, tpr, _ = roc_curve(yte, scores[name])
    ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(yte, scores[name]):.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guessing")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.legend()
titled(ax, "ROC curves: Logistic Regression against Decision Tree",
       "the closer the curve is to the top left corner, the better the model")
finish(f"{RES}/roc_curves.png")

words = vec.get_feature_names_out()
coefs = log_reg.coef_[0]
order = np.argsort(coefs)
idx = np.concatenate([order[:10], order[-10:]])
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.barh(words[idx], coefs[idx], color=["tab:red"] * 10 + ["tab:green"] * 10)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Weight in the model")
titled(ax, "Words the model uses to tell negative from positive",
       "green words push toward positive, red words toward negative")
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

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(k_values, inertias, marker="o")
axes[0].axvline(optimal_k, color="red", linestyle="--", label=f"Chosen K = {optimal_k}")
axes[0].set_xlabel("Number of clusters (K)")
axes[0].set_ylabel("Inertia (lower = tighter clusters)")
axes[0].legend()
titled(axes[0], "Elbow Method", "choose K where the curve stops dropping steeply")
axes[1].plot(k_values[1:], sils[1:], marker="o", color="tab:green")
axes[1].axvline(optimal_k, color="red", linestyle="--")
axes[1].set_xlabel("Number of clusters (K)")
axes[1].set_ylabel("Silhouette Score")
titled(axes[1], "Silhouette Score for each K", "higher means better separated clusters")
finish(f"{RES}/elbow_plot.png")

# ==== 3. Does preparing the features help? ====
versions = {
    "Raw features": cl_raw.values,
    "+ scaled": StandardScaler().fit_transform(cl_raw),
    "+ scaled and clipped": X_final,
}
rows = []
for name, X in versions.items():
    km = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10).fit(X)
    var = pd.DataFrame(X).var()
    rows.append({
        "Version": name,
        "Silhouette": silhouette_score(X_final, km.labels_, sample_size=10000, random_state=RANDOM_STATE),
        "Davies-Bouldin": davies_bouldin_score(X_final, km.labels_),
        "Calinski-Harabasz": calinski_harabasz_score(X_final, km.labels_),
        "Largest column share of variance %": var.max() / var.sum() * 100,
    })
cluster_prep = pd.DataFrame(rows).set_index("Version")
log(f"\nClustering quality for each version of the features (K = {optimal_k}), all judged in the same standardized space:")
log("Higher is better for Silhouette and Calinski-Harabasz, lower is better for Davies-Bouldin.")
log(cluster_prep.round(4).to_string())
log("In the raw version one column carries most of the variance, so K-Means mostly clusters on that single column.")
cluster_prep.to_csv(f"{RES}/clustering_preparation_comparison.csv")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
names = [n.replace(" and ", "\nand ") for n in cluster_prep.index]
for ax, m, hint in zip(axes, ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"],
                       ["higher is better", "lower is better", "higher is better"]):
    ax.bar(names, cluster_prep[m], color=["tab:red", "tab:orange", "tab:green"])
    titled(ax, m, hint)
    for i, v in enumerate(cluster_prep[m]):
        ax.text(i, v, f"{v:.2f}" if m != "Calinski-Harabasz" else f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
fig.suptitle("K-Means: effect of preparing the features (judged in the same standardized space)", y=1.02)
finish(f"{RES}/clustering_preparation_comparison.png")

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

fig, ax = plt.subplots(figsize=(7.5, 7.5))
for c in range(optimal_k):
    members = cl["cluster"] == c
    ax.scatter(cl.loc[members, "longitude"], cl.loc[members, "latitude"], s=3, alpha=0.6, label=f"Cluster {c}")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend(markerscale=4)
titled(ax, f"K-Means clusters on the map (K = {optimal_k})", "each color is one group of similar listings")
finish(f"{RES}/clusters.png")

fig, ax = plt.subplots(figsize=(7.5, 5))
sns.boxplot(data=cl, x="cluster", y="price", ax=ax)
ax.set_yscale("log")
ax.set_xlabel("Cluster")
ax.set_ylabel("Price per night (USD, log scale)")
titled(ax, "Price inside each cluster",
       "the price was not used to make the clusters, so different prices mean the clusters found real differences")
finish(f"{RES}/cluster_prices.png")

# ==== Summary: what did the cleaning change? ====
reg_raw, reg_fin = mean_std(reg_eval, [], "RMSE"), mean_std(reg_eval, chosen_reg, "RMSE")
txt_raw, txt_fin = mean_std(txt_eval, [], "F1"), mean_std(txt_eval, chosen_txt, "F1")
cl_raw_s, cl_fin_s = cluster_prep["Silhouette"].iloc[0], cluster_prep["Silhouette"].iloc[-1]

items = [
    ("Regression: RMSE in USD", "lower is better", reg_raw, reg_fin),
    ("Classification: F1 score", "higher is better", txt_raw, txt_fin),
    ("Clustering: Silhouette Score", "higher is better", (cl_raw_s, 0.0), (cl_fin_s, 0.0)),
]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 5))
summary_rows = []
for ax, (title_, hint, raw, fin) in zip(axes, items):
    ax.bar(["Before\n(raw data)", "After\n(chosen cleaning)"], [raw[0], fin[0]],
           yerr=[raw[1], fin[1]], capsize=4, color=["tab:red", "tab:green"])
    change = fin[0] - raw[0]
    pct = change / abs(raw[0]) * 100 if raw[0] else 0
    titled(ax, title_, f"{hint}; change {change:+.3f} ({pct:+.1f}%)")
    for i, v in enumerate([raw[0], fin[0]]):
        ax.text(i, v, f"{v:.3f}" if v < 10 else f"{v:.1f}", ha="center", va="bottom", fontsize=10)
    summary_rows.append({"task": title_, "before": raw[0], "after": fin[0], "change": change, "change_%": pct})
fig.suptitle("Before and after cleaning (bars show the mean over 5 splits, lines show the standard deviation)", y=1.02)
finish(f"{RES}/before_after_summary.png")

summary_table = pd.DataFrame(summary_rows).set_index("task")
log("\n===== Summary: before and after cleaning =====")
log(summary_table.round(4).to_string())
summary_table.to_csv(f"{RES}/before_after_summary.csv")

# ==== Save all numbers ====
with open(f"{RES}/metrics.txt", "w") as f:
    f.write("\n".join(log_lines))
print("\nAll numbers were saved to results/metrics.txt")
