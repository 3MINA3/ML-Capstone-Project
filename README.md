# ML Capstone Project

This project walks through the full machine learning pipeline: data acquisition, preprocessing, modeling with three different algorithm types, evaluation, and reporting.

The full write-up is in the final report: [project_report.pdf](project_report.pdf)

Notebook on Google Colab: [Open in Colab](https://colab.research.google.com/drive/1QO3u5DIa-m1bJEQIAy9iL9K7HjiItT1A)

## What the project does

| Task | Model | Dataset | Target |
|---|---|---|---|
| Regression | Linear Regression | California Housing | `median_house_value` |
| Classification | Logistic Regression | EmpatheticDialogues | positive (1) or negative (0) emotion |
| Clustering | K-Means | California Housing (without the label) | none |

### Datasets

- **California Housing**: 20640 districts with 10 columns, from [Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices). Used for regression and clustering.
- **EmpatheticDialogues**: about 25000 short conversations, each labeled with one of 32 emotions, from [Hugging Face](https://huggingface.co/datasets/Estwld/empathetic_dialogues_llm). Used for classification. The text of the `situation` column is used to predict whether the emotion is positive or negative. The 32 emotions are grouped into positive and negative, and four ambiguous emotions (`surprised`, `anticipating`, `sentimental`, `nostalgic`) are left out.

## Repository structure

```
ML-Capstone-Project/
├── data/
│   ├── housing.csv
│   └── empathetic.csv
├── notebooks/
│   └── capstone.ipynb
├── results/
├── README.md
└── project_report.pdf
```

- `data/` holds the two datasets.
- `notebooks/capstone.ipynb` holds all the code.
- `results/` holds the figures, `metrics.txt` with every number, and `cluster_summary.csv`.

## How to run

### Option 1: Google Colab

1. Open the Colab link at the top of this file.
2. In the cell called `Get the data`, replace `3MINA3` with the GitHub username that owns this repository.
3. Choose **Runtime > Run all**.

### Option 2: On your own computer

1. Clone the repository and go into it:

```bash
git clone https://github.com/3MINA3/ML-Capstone-Project.git
cd ML-Capstone-Project
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows, activate it with `venv\Scripts\activate`.

3. Install the libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter datasets
```

4. Start Jupyter, open `notebooks/capstone.ipynb`, and run all cells:

```bash
jupyter notebook
```

The notebook finds the `data/` and `results/` folders by itself when it is opened from the `notebooks/` folder.

## Main steps

1. **Loading and inspection**: size, column types, and the target variable.
2. **Cleaning**: missing values are filled with the median, text columns are One-Hot Encoded, and box plots are used to look for outliers. Districts whose price was capped at 500001 are removed, because their real price is unknown.
3. **Scaling and splitting**: 80% train and 20% test. `StandardScaler` is fitted on the training set only. The text is turned into numbers with `TfidfVectorizer`.
4. **Modeling and evaluation**: the three models above, each with its own metrics.

## Results

| Task | Metric | Value |
|---|---|---|
| Linear Regression | RMSE | 62562 USD |
| Linear Regression | R2 | 0.6075 |
| Logistic Regression | Accuracy | 0.8488 |
| Logistic Regression | Precision | 0.8436 |
| Logistic Regression | Recall | 0.7906 |
| K-Means | Chosen K (Elbow Method) | 5 |
| K-Means | Silhouette Score | 0.2816 |

Before the capped districts were removed, the regression gave an RMSE of 70060 USD and an R2 of 0.6254. The R2 is lower after the removal because the prices became less spread out, while the error in dollars went down. The two values cannot be compared directly, since the data and the test set changed.

For the confusion matrix, the precision and recall trade-off, and the interpretation of the clusters, see the [final report](project_report.pdf).

## Libraries

`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`, `datasets`
