import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


def main():
    df = pd.read_csv("nsl_landmarks.csv")
    df_clean = df.drop_duplicates()

    X = df_clean.drop("label", axis=1).values
    y = df_clean["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=300
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    labels = np.unique(y_test)
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    misclass_rows = []
    for i, true_label in enumerate(labels):
        true_count = cm[i].sum()
        if true_count == 0:
            continue
        for j, pred_label in enumerate(labels):
            if i == j:
                continue
            count = int(cm[i, j])
            if count == 0:
                continue
            pct = (count / true_count) * 100
            misclass_rows.append(
                {
                    "true_label": int(true_label),
                    "predicted_label": int(pred_label),
                    "count": count,
                    "percentage": pct
                }
            )

    misclass_df = pd.DataFrame(misclass_rows)
    misclass_df = misclass_df.sort_values(by=["count", "percentage"], ascending=False)

    output_dir = os.path.join("outputs")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "static_failure_cases.csv")
    misclass_df.to_csv(csv_path, index=False)

    top_n = min(10, len(misclass_df))
    if top_n > 0:
        top_df = misclass_df.head(top_n)
        labels_text = [
            f"{row.true_label}->{row.predicted_label}"
            for row in top_df.itertuples(index=False)
        ]

        plt.figure(figsize=(9, 4.5))
        plt.bar(labels_text, top_df["count"])
        plt.ylabel("Count")
        plt.title("Top Static Misclassifications")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        fig_path = os.path.join(output_dir, "static_failure_cases.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
    else:
        fig_path = None

    print(f"Saved static failure cases CSV: {csv_path}")
    if fig_path:
        print(f"Saved static failure cases plot: {fig_path}")


if __name__ == "__main__":
    main()
