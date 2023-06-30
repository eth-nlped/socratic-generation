import copy
import os

import torch
from dataloader import get_input_output_ques_gen_seq, get_input_output_qna_seq, read_jsonl, text_to_digits
import csv
from tqdm.auto import tqdm

from model import DialogueGenerator
from util import initialize_config, deterministic_behaviour, initialise_tokenizer
import sys


def get_input_samples(split: str):
    """Get input questions based on the split

    Args:
        split (str): split : train or test

    Returns:
        input_seq [list]: List of all questions
    """
    all_data = read_jsonl(split)
    input_sequences = [data["question"] for data in all_data]
    input_sentences = copy.deepcopy(input_sequences)

    for iteration in range(len(input_sentences)):
        input_sentences[iteration] = text_to_digits(input_sentences[iteration])

    return input_sentences


def write_to_csv(model_ckpt_path: str, in_seq: list, out_seq: list, predicted_out: list):
    """Write input, ground truth and prediction to the CSV file
    """
    with open(f"{model_ckpt_path}/test_pred.csv", "w") as pred_file:
        write = csv.writer(pred_file)
        write.writerow(["Question", "Ground Truth", "Prediction"])
        for in_samp, out_samp, pred_samp in zip(in_seq, out_seq, predicted_out):
            write.writerow([in_samp, out_samp, "?\n".join(pred_samp.split("?"))])
    print(f"File saved at :{model_ckpt_path}/test_pred.csv")


class Run:
    def __init__(self, config_name, deterministic=True):
        self.config = initialize_config(config_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if deterministic:
            deterministic_behaviour()

    def load_ckpt(self, model_path, model_ckpt_path):

        """Load tokenizer and the model
        """
        tokenizer = initialise_tokenizer(model_path)
        model = DialogueGenerator(model_name=model_ckpt_path, tokenizer=tokenizer, max_decode_len=100)
        model.to(self.device)
        model.eval()
        return tokenizer, model

    def model_prediction(self, model_path: str, model_ckpt_path: str, in_seq: list):

        """Take model checkpoints and input questions and generate the output
        """

        tokenizer, model = Run.load_ckpt(self, model_path, model_ckpt_path)
        predicted_out = []
        pbar = tqdm(range(len(in_seq)))
        for in_sample in in_seq:
            encoded_sent = tokenizer(in_sample, return_tensors='pt').to(self.device)
            input_ids, attention_mask = encoded_sent.input_ids, encoded_sent.attention_mask
            decoded_out = model.generate(src_input=input_ids, src_mask=attention_mask)
            predicted_out.append(' '.join(decoded_out))
            pbar.update(1)

        return predicted_out

    def test(self):
        conf = self.config

        model_path = os.path.join(conf['PRETRAINED_MODEL_PREFIX_PATH'], conf['MODEL_NAME'])
        model_ckpt_path = conf['MODEL_IMPORT_PATH']
        in_seq, out_seq, data_points_indices = [], [], None

        if conf['TASK'] == "question-answering":
            in_seq, out_seq = get_input_output_qna_seq(conf['SPLIT'])

        elif conf['TASK'] == "question-generation":
            in_seq, out_seq, data_points_indices = get_input_output_ques_gen_seq(conf['ITERATIVE'], conf['SPLIT'],
                                                                                 conf['PLANNING'], conf["REWARD"])

            if conf["CRITIC"] or os.environ.get("CRITIC"):
                model_pre_ckpt_path = conf['CRITIC_MODEL_PATH']
                input_questions = get_input_samples(conf["SPLIT"])
                print(f"Replacing inputs from a critic model: {conf['CRITIC_MODEL_PATH']}")
                prediction = Run.model_prediction(self, model_path, model_pre_ckpt_path, input_questions)
                # Replace input with critic generated sentences
                in_seq = [seq_ques + " [SEP]" + seq_op for seq_ques, seq_op in zip(input_questions, prediction)]
                _, out_seq, _ = get_input_output_ques_gen_seq(conf['SPLIT'], conf['PLANNING'],
                                                              conf["REWARD"], conf["REWARD"])

            else:
                in_seq, out_seq, data_points_indices = get_input_output_ques_gen_seq(conf['ITERATIVE'], conf['SPLIT'],
                                                                                     conf['PLANNING'], conf["REWARD"])

        print("Predicting outputs ....")
        predicted_out = Run.model_prediction(self, model_path, model_ckpt_path, in_seq)

        # Change path for output
        if conf["CRITIC"] or os.environ.get("CRITIC"):
            output_path = os.path.join(model_ckpt_path, "critic")
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
        else:
            output_path = model_ckpt_path

        if conf["write_prediction"]:
            write_to_csv(output_path, in_seq, out_seq, predicted_out)
            if data_points_indices:
                with open(f"{output_path}/test_indices.txt", 'w') as file:
                    file.write('\n'.join([str(x) for x in data_points_indices]))

        return output_path


if __name__ == "__main__":
    config_name = sys.argv[1]
    runner = Run(config_name)
    runner.test()
