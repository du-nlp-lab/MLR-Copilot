import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_score(submission_folder = "../env"):
    submission_path = os.path.join(submission_folder, "submission.csv")
    submission = pd.read_csv(submission_path, index_col=0)
    preds = submission["label"].tolist()
    preds = [float(pred) for pred in preds] 
    lang = "eng"

    test_data_path = os.path.join(submission_folder, "data", lang, f"{lang}_test.csv")
    df = pd.read_csv(test_data_path)
    scores = df["label"].tolist()
    scores = [float(score) for score in scores]    

    spearman_corr = np.corrcoef(scores, preds)[0, 1]
    return spearman_corr

if __name__ == "__main__":
    print(get_score())