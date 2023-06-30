import re
import torch
from nltk.translate.bleu_score import sentence_bleu
import copy
import os
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from calculator import sample


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def return_all_ans(text):
    all_equations = re.findall("<<\S+>>", text)
    return [float(eq.split("=")[-1].split(">>")[0]) for eq in all_equations]


def compute_sentence_bleu(reference_text, pred_text):
    references = [reference_text.strip().split()]
    hypothesis = pred_text.strip().split()
    return sentence_bleu(references, hypothesis)


def bleu_reward_estimation(tokenized_reference_text_list, pred_text_list, repeat):
    tokenized_reference_list = copy.deepcopy(tokenized_reference_text_list * repeat)
    batch_size = len(tokenized_reference_list)
    sentence_bleu_list = []
    for k in range(batch_size):
        one_ref_sen = tokenized_reference_list[k]
        one_pred_sen = pred_text_list[k]
        one_sen_bleu = compute_sentence_bleu(one_ref_sen, one_pred_sen)
        sentence_bleu_list.append(one_sen_bleu)

    return sentence_bleu_list


def correct_ques_num_reward_estimation(tokenized_reference_text_list, pred_text_list, repeat):
    tokenized_reference_text_list = copy.deepcopy(tokenized_reference_text_list * repeat)
    tokenized_pred_text_list = []
    for text in pred_text_list:
        if len(re.findall("\?", ' '.join(text))) != 0:
            tokenized_pred_text_list.append(len(re.findall("\?", ' '.join(text))))
        else:
            tokenized_pred_text_list.append(1)

    # Normalised reward based on formula : 1 - (|num_of_pred_ques - gt_ques| / gt_ques)
    batch_size = len(tokenized_reference_text_list)
    correct_ques_list_reward = []
    for k in range(batch_size):
        gt_ques_num = len(re.findall("\?", tokenized_reference_text_list[k]))
        reward = 1 - (abs(gt_ques_num - tokenized_pred_text_list[k]) / max(gt_ques_num, 1))
        correct_ques_list_reward.append(reward)

    return correct_ques_list_reward

def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def qa_reward_estimation(input_text_list, pred_text_list, repeat, qa_model_path, export_path, epoch,
                         partial_reward=False):
    device = torch.device("cuda")
    tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(qa_model_path, "gpt-2-tokenizer"))
    model = GPT2LMHeadModel.from_pretrained(qa_model_path)
    model.to(device)

    input_main_ques = copy.deepcopy(input_text_list * repeat)
    batch_size = len(input_main_ques)
    overall_correct_ans = []

    # To be changed in future as cannot be hard-coded
    answer_file_path = os.path.join("data/train.jsonl")
    data = read_jsonl(answer_file_path)

    for k in range(batch_size):
        one_ref_sen = input_main_ques[k].split("[SEP]")[0]
        one_pred_sen = pred_text_list[k]
        qn = one_ref_sen + " <<" + one_pred_sen + ">>/n"
        sample_len = 128
        prediction, step_wise_ans = sample(model, qn, tokenizer, device, sample_len)

        answer = "<not-found>"
        for samples in data:
            if one_ref_sen.strip() == samples["question"].strip():
                answer = samples["answer"] + " <|endoftext|>"

        with open(os.path.join(export_path, "qa_rewards.txt"), "a") as tempfile:
            tempfile.write(str(epoch) + "\t" + prediction + "\t" + "gt: " + answer + "\t")

        if partial_reward:
            all_gt_ans = return_all_ans(answer)
            overall_correct_ans.append(
                len([value for value in all_gt_ans if value in step_wise_ans]) / max(len(all_gt_ans), 1))

        else:
            gt_answer = extract_answer(answer)

            model_ans = extract_answer(prediction)
            if gt_answer == model_ans:
                overall_correct_ans.append(1)
            else:
                overall_correct_ans.append(0)

    return overall_correct_ans
