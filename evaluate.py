import os
from typing import List
import matplotlib.pyplot as plt

import nltk
import seaborn as sns
import pandas as pd
import sys
import numpy as np
from sacrebleu.metrics import BLEU
from sentence_transformers import SentenceTransformer, util
from bert_score import score

GROUND_TRUTH_COLUMN = "Ground Truth"
PREDICTION_COLUMN = "Prediction"


def compute_corpus_bleu(ground_truths: List[str], predictions: List[str]):
    assert len(ground_truths) == len(predictions)
    list_of_target_token_list, list_of_pred_token_list = [], []
    for k in range(len(ground_truths)):
        one_target_token_list = ground_truths[k].strip().split()
        list_of_target_token_list.append(one_target_token_list)
        one_pred_token_list = predictions[k].strip().split()
        list_of_pred_token_list.append(one_pred_token_list)

    list_of_references, list_of_hypotheses = [], []
    for k in range(len(list_of_target_token_list)):
        list_of_references.append([list_of_target_token_list[k]])
        list_of_hypotheses.append(list_of_pred_token_list[k])
    return nltk.translate.bleu_score.corpus_bleu(list_of_references,
                                                 list_of_hypotheses, weights=[(1, 0.0, 0.0, 0.0),
                                                                              (1. / 2., 1. / 2.),
                                                                              (1. / 3., 1. / 3., 1. / 3.),
                                                                              (1. / 4., 1. / 4., 1. / 4., 1. / 4.)])


def compute_bert_score(ground_truths: List[str], predictions: List[str]):
    assert len(ground_truths) == len(predictions)

    # define model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    gt_embeddings = model.encode(ground_truths, convert_to_tensor=True)
    pred_embeddings = model.encode(predictions, convert_to_tensor=True)

    cosine_score = [util.cos_sim(gt_embed, pred_embed).item() for gt_embed, pred_embed in
                    zip(gt_embeddings, pred_embeddings)]
    return (sum(cosine_score) / len(ground_truths)), cosine_score


def simple_count_number_of_questions(text: str) -> int:
    return text.count("?")


def save_results(df: pd.DataFrame, path: str):
    filename = os.path.join(path, "evaluation.csv")
    df.reset_index(inplace=True)
    df.to_csv(filename)
    print(f"File saved at :{filename}")


def main(PATH):
    df = pd.read_csv(PATH)
    all_gt = list(df[GROUND_TRUTH_COLUMN])
    all_pred = list(df[PREDICTION_COLUMN])

    sentence_bleu = []
    for sample_gt, sample_pred in zip(all_gt, all_pred):
        reference = sample_gt.strip().split()
        hypothesis = sample_pred.strip().split()
        BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis,
                                                            weights=[(1, 0.0, 0.0, 0.0),
                                                                     (1. / 2., 1. / 2.),
                                                                     (1. / 3., 1. / 3., 1. / 3.),
                                                                     (1. / 4., 1. / 4., 1. / 4., 1. / 4.)])
        sentence_bleu.append(BLEUscore)

    df["s_bleu1"], df["s_bleu2"], df["s_bleu3"], df["s_bleu4"] = zip(*list(sentence_bleu))
    df["gt_questions_count"] = df[GROUND_TRUTH_COLUMN].apply(lambda x: simple_count_number_of_questions(x))
    df["prediction_questions_count"] = df[PREDICTION_COLUMN].apply(lambda x: simple_count_number_of_questions(x))

    print(
        f"Correct question numbers prediction: {((df['gt_questions_count'] - df['prediction_questions_count']) == 0).sum()} out of {len(df['prediction_questions_count'])} or {(((df['gt_questions_count'] - df['prediction_questions_count']) == 0).sum()) / len(df['prediction_questions_count'])}")
    sentence_bleu = np.array(sentence_bleu)
    print(f"Overall sentence BLUE score: {np.average(sentence_bleu, axis=0) * 100}")

    corpus_bleu = compute_corpus_bleu(all_gt, all_pred)
    print(f"Overall corpus BLUE score: {corpus_bleu * np.array([100, 100, 100, 100])}")

    overall_bert_score, bert_score = compute_bert_score(all_gt, all_pred)
    df["bert_score"] = bert_score
    print(f"Overall BERT score: {overall_bert_score}")

    P, R, F1 = score(all_pred, all_gt, lang="en", model_type="microsoft/deberta-xlarge-mnli", verbose=False)
    print(f"System level F1 score: {F1.mean():.10f}")

    sacre_bleu = BLEU()
    sacrebleu_scores = sacre_bleu.corpus_score(all_pred, [all_gt]).format()
    print(f"Sacrebleu results: {sacrebleu_scores}")
    print(sacre_bleu.get_signature())

    save_results(df, os.path.dirname(PATH))

    fig, axs = plt.subplots(4, 1, figsize=(15, 8))
    sns.histplot(data=df, x="gt_questions_count", label="Ground truth questions", color="green", alpha=0.2, ax=axs[0])
    sns.histplot(data=df, x="prediction_questions_count", label="Prediction questions", color="blue", alpha=0.2,
                 ax=axs[0])
    sns.boxplot(data=df, x="gt_questions_count", y="s_bleu4", ax=axs[1])
    sns.histplot(data=df, x="bert_score", ax=axs[2])
    sns.boxplot(data=df, x="gt_questions_count", y="bert_score", ax=axs[3])
    axs[0].legend()
    plt.savefig(os.path.join(os.path.dirname(PATH), "distribution.png"))


if __name__ == "__main__":
    PATH = sys.argv[1]
    main(PATH)
