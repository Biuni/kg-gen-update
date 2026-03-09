import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# load csv
def load_dataset(path):

    df = pd.read_csv(path)

    y = df["label"]

    non_features = ["run_id", "label"]
    features = [c for c in df.columns if c not in non_features]

    X = df[features]

    X = X.select_dtypes(include=np.number)
    X = X.fillna(0)

    return df, X, y


# Basic Stats
def basic_stats(df):

    print("\n=== BASIC DATASET INFO ===")
    print("Number of runs:", len(df)) # numero di campioni

    print("\nLabel distribution:")
    print(df["label"].value_counts()) # numero di campioni per tipo di risultato

    print("\nDataset summary:")
    print(df.describe()) # summary


# Stats per Label
def stats_per_label(df, features):

    print("\n=== MEAN FEATURES PER LABEL ===")

    means = df.groupby("label")[features].mean() # media di ogni feature per tipo di risultato
    print(means)

    print("\n=== STD FEATURES PER LABEL ===")

    stds = df.groupby("label")[features].std() # deviazione standard di ogni feature per tipo di risultato
    print(stds)

    return means, stds


# ANOVA
def compute_anova(X, y):

    """
    Perform ANOVA (Analysis of Variance) to evaluate the discriminative
    power of each feature with respect to the class labels.

    ANOVA is a statistical test that measures whether the mean values of
    a feature differ significantly across multiple groups (here: the labels/results).

    In this context:
        - each feature is tested independently
        - the groups correspond to the different labels (e.g., agent invalid action_false,
          completed_false, completed_true...)

    The test compares two sources of variance:

        1) Between-group variance
           How different the feature means are across labels.

        2) Within-group variance
           How much the feature varies inside each label.

    The ANOVA F-score is defined as:

        F = (variance_between_groups) / (variance_within_groups)

    Interpretation:
        - High F-score → the feature has very different means across labels
                         relative to its internal variability.
                         This suggests strong discriminative power.

        - Low F-score → the feature behaves similarly across labels
                        and is likely not useful for distinguishing classes.

    The test also produces a p-value, which indicates the statistical
    significance of the observed difference between groups.

    Interpretation of p-value:
        - p < 0.05  → statistically significant difference between labels
        - p < 0.01  → strong evidence of difference
        - p >= 0.05 → differences may be due to random variation

    In this script:
        - ANOVA is used as a feature ranking method.
        - Features are sorted by F-score (highest first).

    Therefore:
        Features with the highest F-scores are considered the most
        discriminative with respect to the target labels.

    Parameters
    ----------
    X : DataFrame
        Feature matrix (samples × features).

    y : array-like
        Encoded class labels.

    Returns
    -------
    res : DataFrame
        Table containing:
            - feature name
            - ANOVA F-score
            - p-value
        sorted by descending F-score.
    """

    print("\n=== ANOVA FEATURE IMPORTANCE ===")

    f_vals, p_vals = f_classif(X, y)

    res = pd.DataFrame({
        "feature": X.columns,
        "f_score": f_vals,
        "p_value": p_vals
    })

    res = res.sort_values("f_score", ascending=False)

    print(res.head(15))

    return res


# MUTUAL INFORMATION
def compute_mutual_information(X, y):
    """
    Calcola l'importanza delle feature rispetto alla label usando la Informazione Mutua (Mutual Information).

    La Mutual Information (MI) misura quanta informazione conosciuta su una variabile (feature) riduce l'incertezza su
    un'altra variabile (la label). In altre parole, valuta quanto una feature è informativa
    riguardo alla classificazione delle run.

    Dettagli e interpretazione:
    - MI >= 0 sempre; valori più alti indicano una maggiore correlazione non lineare tra la feature e la label.
    - MI = 0 significa che la feature non fornisce alcuna informazione sulla label (indipendenza statistica).
    - Non esiste un massimo teorico fisso; dipende dall'entropia della variabile target. Si interpreta confrontando
      le MI relative tra feature dello stesso dataset.
    - A differenza dell'ANOVA (che misura solo differenze lineari tra medie), la MI può catturare relazioni non lineari
      e non gaussiane.

    Procedura:
    1. mutual_info_classif(X, y) calcola la MI tra ciascuna feature numerica e la label categoriale y.
    2. Viene creato un DataFrame con colonne:
       - 'feature': il nome della feature
       - 'mutual_info': il valore di MI calcolato
    3. Le feature vengono ordinate in modo decrescente in base alla MI, così da vedere subito quali feature
       forniscono più informazioni sulla label.

    Returns:
    - res: DataFrame con feature e rispettiva mutual information, ordinato
      per importanza decrescente.
    """

    print("\n=== MUTUAL INFORMATION ===")

    mi = mutual_info_classif(X, y)

    res = pd.DataFrame({
        "feature": X.columns,
        "mutual_info": mi
    })

    res = res.sort_values("mutual_info", ascending=False)

    print(res.head(15))

    return res


# RANDOM FOREST IMPORTANCE
def compute_random_forest_importance(X, y):
    """
    Calcola l'importanza delle feature usando un modello Random Forest.

    Random Forest è un ensemble di alberi decisionali. La "feature importance"
    misura quanto ciascuna feature contribuisce alla riduzione dell'impurità
    (ad esempio Gini o entropia) su tutti gli alberi del modello.

    Procedura:
    1. Il dataset viene diviso in training (80%) e test (20%).
    2. Si addestra una Random Forest con 500 alberi sul training set.
    3. Si calcola l'accuratezza sul test set come indicatore generale
       della bontà del modello.
    4. Si estraggono le feature importances fornite dal modello:
       - Valori tra 0 e 1, con somma totale = 1.
       - Più alto è il valore, più quella feature contribuisce alla
         decisione degli alberi nel classificare correttamente le label.
    5. Si costruisce un DataFrame con le feature ordinate per importanza decrescente.

    Interpretazione:
    - Una feature con importance = 0.25 indica che circa il 25% della
      riduzione complessiva dell'impurità durante l'addestramento
      è attribuibile a quella feature.
    - Features con valori molto bassi o 0 sono poco rilevanti per la
      predizione.

    Differenze rispetto ad ANOVA o Mutual Information:
    - Random Forest cattura relazioni lineari e non lineari tra feature e label.
    - Tiene conto dell'interazione tra più feature.
    - Non fornisce valori statistici (p-value) come ANOVA, ma un'indicazione
      pratica di utilità per la predizione.

    Returns:
    - res: DataFrame con colonne:
        'feature': nome della feature
        'importance': valore di importanza della feature
      ordinato per importanza decrescente.
    """

    print("\n=== RANDOM FOREST FEATURE IMPORTANCE ===")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=500,
        random_state=42
    )

    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)

    print("RandomForest accuracy:", acc)

    res = pd.DataFrame({
        "feature": X.columns,
        "importance": clf.feature_importances_
    })

    res = res.sort_values("importance", ascending=False)

    print(res.head(15))

    return res


# -----------------------------
# EFFECT SIZE
# -----------------------------

def pairwise_effect_size(df, features):
    """
    Calcola l'effetto di Cohen (Cohen's d) tra tutte le coppie di label per ciascuna feature numerica del dataset.

    L'effetto di Cohen misura la grandezza dell'effetto tra due gruppi, ed è definito come la differenza tra le medie dei
    gruppi divisa per la deviazione standard combinata.  

    Formula:
        d = (mean1 - mean2) / sqrt((std1^2 + std2^2)/2)

    Parametri:
    - df: DataFrame contenente le feature numeriche e la colonna 'label'
    - features: lista di feature numeriche su cui calcolare l'effetto

    Procedura:
    1. Individua tutte le label uniche nel dataset.
    2. Per ciascuna feature, calcola Cohen's d per ogni coppia di label.
    3. Se entrambe le label hanno deviazione standard = 0, ignora (nessuna variazione).
    4. Memorizza la feature, le due label e il valore assoluto di Cohen's d.

    Interpretazione:
    - Cohen's d fornisce una misura della separazione dei gruppi per una feature:
        * 0.2 ≈ effetto piccolo
        * 0.5 ≈ effetto medio
        * 0.8 ≈ effetto grande
    - Più alto è d, più la feature distingue bene le due label.
    - Usare il valore assoluto permette di considerare la differenza indipendentemente dalla direzione.

    Returns:
    - res: DataFrame con colonne:
        'feature' : nome della feature
        'label_a' : prima label della coppia
        'label_b' : seconda label della coppia
        'cohen_d' : valore assoluto di Cohen's d tra i due gruppi
      ordinato in ordine decrescente di cohen_d, mostrando prima le feature più discriminanti.
    """

    print("\n=== PAIRWISE EFFECT SIZE (COHEN D) ===")

    labels = df["label"].unique()

    results = []

    for f in features:

        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):

                l1 = labels[i]
                l2 = labels[j]

                x1 = df[df.label == l1][f]
                x2 = df[df.label == l2][f]

                if x1.std() == 0 and x2.std() == 0:
                    continue

                d = (x1.mean() - x2.mean()) / np.sqrt(
                    (x1.std()**2 + x2.std()**2) / 2
                )

                results.append({
                    "feature": f,
                    "label_a": l1,
                    "label_b": l2,
                    "cohen_d": abs(d)
                })

    res = pd.DataFrame(results)

    res = res.sort_values("cohen_d", ascending=False)

    print(res.head(20))

    return res


# -----------------------------
# RUN ANALYSIS FOR ONE DATASET
# -----------------------------

def run_analysis(dataset_path, output_dir):

    print(f"\n\n=== ANALYZING {dataset_path.name} ===")

    df, X, y = load_dataset(dataset_path)

    features = list(X.columns)

    basic_stats(df)

    means, stds = stats_per_label(df, features)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    anova = compute_anova(X, y_enc)
    mi = compute_mutual_information(X, y_enc)
    rf = compute_random_forest_importance(X, y_enc)
    effect = pairwise_effect_size(df, features)

    # save results
    means.to_csv(output_dir / "mean_per_label.csv")
    stds.to_csv(output_dir / "std_per_label.csv")
    anova.to_csv(output_dir / "anova_results.csv", index=False)
    mi.to_csv(output_dir / "mutual_information.csv", index=False)
    rf.to_csv(output_dir / "rf_importance.csv", index=False)
    effect.to_csv(output_dir / "effect_size.csv", index=False)


# -----------------------------
# MAIN
# -----------------------------

def main():

    base_dir = Path(__file__).parent

    results_dir = base_dir / "results"

    stats_dir = base_dir / "stats"
    motifs_dir = stats_dir / "motifs"
    patterns_dir = stats_dir / "patterns"
    global_dir = stats_dir / "global_metrics"

    motifs_dir.mkdir(parents=True, exist_ok=True)
    patterns_dir.mkdir(parents=True, exist_ok=True)
    global_dir.mkdir(parents=True, exist_ok=True)

    motifs_path = results_dir / "motifs_counts.csv"
    patterns_path = results_dir / "missing_patterns.csv"
    global_path = results_dir / "global_metrics.csv"

    run_analysis(motifs_path, motifs_dir)
    run_analysis(patterns_path, patterns_dir)
    run_analysis(global_path, global_dir)


if __name__ == "__main__":
    main()