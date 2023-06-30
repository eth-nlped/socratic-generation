import os
import time
import sys

import torch
import wandb

from transformers import get_scheduler, AdamW
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from dataloader import read_jsonl, critic_data_prep, GSMDataset
from model import DialogueGenerator
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
        model = DialogueGenerator(model_name=model_path, tokenizer=tokenizer, max_decode_len=100)
        model.to(self.device)

        if conf['TASK'] == "question-generation":
            in_seq, out_seq = critic_data_prep(conf['SPLIT'], conf['PLANNING'])
        else:
            raise NotImplementedError

        train_dset = GSMDataset(tokenizer, in_seq, out_seq)
        train_loader = DataLoader(train_dset, batch_size=int(conf["BATCH_SIZE"]), shuffle=True, drop_last=True)

        optim = AdamW(model.parameters(), lr=conf["LEARNING_RATE"])
        num_training_steps = int(conf["EPOCHS"]) * len(train_loader)
        lr_scheduler = get_scheduler(
            conf['LR_SCHEDULER'],
            optimizer=optim,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        optim.zero_grad()

        # Add timestamp to the path to not override model exports
        conf['export_path'] = os.path.join(conf['EXPORT_PREFIX_PATH'], conf['MODEL_CKPT_PATH'] + str(time.time_ns()))

        wandb.init(project=conf['TASK'], entity="", config = conf)
        wandb.run.name = f'pre-train-{conf["MODEL_NAME"]}-{conf["planning"]}'
        output_dict ={}
        export_path = ""

        model.train()
        pbar = tqdm(range(num_training_steps))
        for epoch in range(int(conf["EPOCHS"])):
            # Print generated samples for a training example after every epoch to see the progress.
            test_examples = read_jsonl("test")
            qn = test_examples[2]["question"]
            encoded_sent = tokenizer(qn, return_tensors='pt').to(self.device)
            input_ids, attention_mask = encoded_sent.input_ids, encoded_sent.attention_mask
            output_dict[epoch] = ' '.join(model.generate(src_input=input_ids, src_mask=attention_mask))

            model.train()
            for batch in train_loader:
                optim.zero_grad()
                for k, v in batch.items():
                    if k != "raw_output":
                        batch[k] = v.to(self.device)
                mle_loss = model(batch["inpt"], batch["att_mask"], batch["lbl"])
                train_loss = mle_loss
                wandb.log({"loss": train_loss})
                train_loss.backward()
                if conf["USE_GRADIENT_CLIPPING"] != "":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                lr_scheduler.step()
            pbar.update(1)

        model.model.save_pretrained(os.path.join(export_path, "final"))
        return export_path


if __name__ == "__main__":
    config_name = sys.argv[1]
    runner = Run(config_name)
    runner.train()
