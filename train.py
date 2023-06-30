import json
import os
import time
import sys
import numpy as np
import torch
import wandb
import pandas as pd
from transformers import get_scheduler, AdamW
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from dataloader import get_input_output_ques_gen_seq, get_input_output_qna_seq, read_jsonl, GSMDataset
from model import DialogueGenerator
from rewards import bleu_reward_estimation, correct_ques_num_reward_estimation, qa_reward_estimation
from util import initialize_config, deterministic_behaviour, initialise_tokenizer


class Run:
    def __init__(self, config_name, deterministic=True):
        self.config = initialize_config(config_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if deterministic:
            deterministic_behaviour()

    def train(self):
        conf = self.config

        model_path = os.path.join(conf['PRETRAINED_MODEL_PREFIX_PATH'], conf['MODEL_NAME'])
        tokenizer = initialise_tokenizer(model_path)
        if conf["MODEL_IMPORT_PATH"]:
            model_path = conf["MODEL_IMPORT_PATH"]
        print(f"Loading model from: {model_path}")
        model = DialogueGenerator(model_name=model_path, tokenizer=tokenizer, max_decode_len=100)
        model.to(self.device)

        data_points_indices = []
        if conf['TASK'] == "question-answering":
            in_seq, out_seq = get_input_output_qna_seq(conf['SPLIT'])
        elif conf['TASK'] == "question-generation":
            in_seq, out_seq, data_points_indices = get_input_output_ques_gen_seq(conf['ITERATIVE'], conf['SPLIT'],
                                                                                 conf['PLANNING'], conf['REWARD'])

        train_dset = GSMDataset(tokenizer, in_seq, out_seq)
        batch_size = int(conf["BATCH_SIZE"])
        train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)

        optim = AdamW(model.parameters(), lr=conf["LEARNING_RATE"])
        num_training_steps = int(conf["EPOCHS"]) * len(train_loader)
        lr_scheduler = get_scheduler(
            conf['LR_SCHEDULER'],
            optimizer=optim,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        optim.zero_grad()

        rl_epochs = int(int(conf["EPOCHS"]) - (float(conf["RL_EPOCHS"]) * int(conf["EPOCHS"])))
        # Add timestamp to the path to not override model exports
        export_path = os.path.join(conf['EXPORT_PREFIX_PATH'], conf['MODEL_CKPT_PATH'] + str(time.time_ns()))
        if not os.path.isdir(export_path):
            os.makedirs(export_path)
        with open(os.path.join(export_path, 'config.json'), 'w') as file:
            json.dump(conf, file)

        wandb.init(project=conf['TASK'], entity="", config=conf)
        wandb.config.model_path = model_path
        wandb.config.rl_epochs = rl_epochs
        wandb.config.export_path = export_path
        wandb.run.name = os.environ.get("RUN_NAME", f'{conf["MODEL_NAME"]}-{conf["PLANNING"]}')
        output_dict = {}

        table = wandb.Table(data=pd.DataFrame({"input": in_seq[:5],
                                               "output": out_seq[:5]}))
        wandb.log({'training_datasample': table})
        epoch_results_table = wandb.Table(columns=["epoch", "prediction"])

        max_valid_score = 0.0
        model.train()
        pbar = tqdm(range(num_training_steps))
        for epoch in range(int(conf["EPOCHS"])):
            # Train loop
            model.train()
            for batch in train_loader:
                optim.zero_grad()
                for k, v in batch.items():
                    if k != "raw_output" and k != "raw_input":
                        batch[k] = v.to(self.device)
                mle_loss = model(batch["inpt"], batch["att_mask"], batch["lbl"])
                train_loss = mle_loss

                # RL loop
                if epoch >= rl_epochs:
                    # Compute RL loss
                    train_gathered_logprobs, train_indicator_matrix, decoded_result_list = model.rl_sampling(
                        batch["inpt"], batch["att_mask"], top_p=conf["TOP_P"], return_seq=int(conf["RETURN_SEQ"]),
                        temperature=conf["TEMPERATURE"], num_beams=conf["NUM_BEAMS"])

                    if 'fluency' in conf["REWARD"].lower() or 'combined' in conf["REWARD"].lower():
                        # measure reward for question generation and correct number of questions
                        question_generation_reward = bleu_reward_estimation(batch["raw_output"], decoded_result_list,
                                                                            int(conf["RETURN_SEQ"]))
                        assert len(question_generation_reward) == batch_size * int(conf["RETURN_SEQ"])

                    if 'number' in conf["REWARD"].lower() or 'combined' in conf["REWARD"].lower():
                        correct_ques_num_reward = correct_ques_num_reward_estimation(batch["raw_output"],
                                                                                     decoded_result_list,
                                                                                     int(conf["RETURN_SEQ"]))
                        assert len(correct_ques_num_reward) == batch_size * int(conf["RETURN_SEQ"])

                    if 'qa' in conf["REWARD"].lower() or 'combined' in conf["REWARD"].lower():
                        qa_reward = qa_reward_estimation(batch["raw_input"], decoded_result_list,
                                                         int(conf["RETURN_SEQ"]), conf["QA_MODEL_PATH"],
                                                         export_path, epoch,
                                                         partial_reward=bool(conf["QA_PARTIAL_REWARD"]))
                        assert len(qa_reward) == batch_size * int(conf["RETURN_SEQ"])

                    # Combination
                    if 'fluency' in conf["REWARD"].lower():
                        # multiply by hyperparameter
                        total_reward = question_generation_reward
                    if 'number' in conf["REWARD"].lower():
                        total_reward = correct_ques_num_reward
                    if 'qa' in conf["REWARD"].lower():
                        total_reward = qa_reward
                    if 'combined' in conf["REWARD"].lower():
                        total_reward = [reward_a + reward_b + reward_c for reward_a, reward_b, reward_c in
                                        zip(question_generation_reward, correct_ques_num_reward, qa_reward)]

                    train_reward = torch.FloatTensor(total_reward).type(train_indicator_matrix.type()).unsqueeze(-1)
                    assert train_reward.size() == torch.Size([batch_size * int(conf["RETURN_SEQ"]), 1])

                    train_sample_logprobs = train_gathered_logprobs * train_indicator_matrix
                    train_RL_term = train_reward * train_sample_logprobs
                    train_RL_loss = (-1 * torch.sum(train_RL_term)) / torch.sum(train_indicator_matrix)

                    alpha, beta = 0.5, 0.5
                    train_loss = alpha * mle_loss + beta * train_RL_loss
                    wandb.log({"rl_loss": train_RL_loss, "mle_loss": mle_loss})

                wandb.log({"loss": train_loss})
                train_loss.backward()
                if conf["USE_GRADIENT_CLIPPING"] != "":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                lr_scheduler.step()
                pbar.update(1)

            # Validation set metrics
            if epoch % int(conf["VALID_EVERY_EPOCH"]) == 0:
                model.eval()
                valid_in_seq, valid_out_seq, _ = get_input_output_ques_gen_seq(conf['ITERATIVE'], 'valid',
                                                                               conf['PLANNING'],
                                                                               conf["REWARD"].lower())
                with torch.no_grad():
                    print(f"({epoch}) Starting validation evaluation...")
                    predicted_out = []
                    for in_sample in valid_in_seq:
                        encoded_sent = tokenizer(in_sample, return_tensors='pt').to(self.device)
                        input_ids, attention_mask = encoded_sent.input_ids, encoded_sent.attention_mask
                        decoded_out = model.generate(src_input=input_ids, src_mask=attention_mask)
                        predicted_out.append(' '.join(decoded_out))
                    valid_bleu_list = bleu_reward_estimation(valid_out_seq, predicted_out, 1)
                    valid_question_count_list = correct_ques_num_reward_estimation(valid_out_seq, predicted_out, 1)
                    mean_valid_bleu = np.array(valid_bleu_list).mean()
                    mean_valid_question_count = np.array(valid_question_count_list).mean()
                    print(f"Valid bleu: {mean_valid_bleu}, valid question count: {mean_valid_question_count}")
                    wandb.log({"valid_bleu": mean_valid_bleu, "valid_question_count": mean_valid_question_count})

                if mean_valid_bleu > max_valid_score:
                    print(f"Saving best model with valid BLEU score {mean_valid_bleu}")
                    model.model.save_pretrained(os.path.join(export_path, "best_valid"))
                    max_valid_score = mean_valid_bleu

                # Print generated samples for a training example after every epoch to see the progress.
                test_examples = read_jsonl("test")
                qn = test_examples[2]["question"]
                encoded_sent = tokenizer(qn, return_tensors='pt').to(self.device)
                input_ids, attention_mask = encoded_sent.input_ids, encoded_sent.attention_mask
                output_dict[epoch] = ' '.join(model.generate(src_input=input_ids, src_mask=attention_mask))
                epoch_results_table.add_data(epoch, output_dict[epoch])

        model.model.save_pretrained(os.path.join(export_path, "final"))
        wandb.log({'after_batch_prediction': epoch_results_table})

        # save intermediate results in a csv file
        with open(f"{export_path}/intermediate_results.csv", 'w') as f:
            for key in output_dict.keys():
                f.write("%s,%s\n" % (key, output_dict[key]))
        if data_points_indices:
            with open(f"{export_path}/train_indices.txt", 'w') as file:
                file.write('\n'.join([str(x) for x in data_points_indices]))
        print("Finish")

        return export_path


if __name__ == "__main__":
    config_name = sys.argv[1]
    runner = Run(config_name)
    runner.train()
