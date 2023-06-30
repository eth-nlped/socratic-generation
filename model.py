from typing import List

import torch
from torch import nn
from transformers import BartForConditionalGeneration, BartConfig, T5Config, T5ForConditionalGeneration


class DialogueGenerator(nn.Module):
    def __init__(self, model_name: str, tokenizer, max_decode_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_decode_len = max_decode_len

        if 't5' in model_name:
            print('Initializing T5 model...')
            t5_config = T5Config.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name, config=t5_config)
        elif 'bart' in model_name:
            print('Initializing BART model...')
            bart_config = BartConfig.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(model_name, config=bart_config)
        else:
            raise NotImplementedError

        print('Resizing Token Embeddings...')
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.vocab_size = len(self.tokenizer)
        self.logsftmax = nn.LogSoftmax(dim=-1)
        self.padding_idx = self.tokenizer.pad_token_id

    def forward(self, src_input, src_mask, tgt_output):
        src_mask = src_mask.type(src_input.type())
        outputs = self.model(input_ids=src_input, attention_mask=src_mask, labels=tgt_output)
        loss = outputs[0]
        return loss

    def generate(self, src_input, src_mask) -> List[str]:
        """
        Greedy decoding.
        Args:
            src_input:
            src_mask:

        Returns:

        """
        result_list = []
        outputs = self.model.generate(input_ids=src_input, attention_mask=src_mask, max_length=self.max_decode_len)
        for predicted_ids in outputs:
            one_result = self.tokenizer.decode(predicted_ids, skip_special_tokens=True)
            result_list.append(one_result)
        return result_list

    def rl_sampling(self, src_input, src_mask, top_p: int, return_seq: int, temperature: float, num_beams: int = 1):
        sample_output = self.model.generate(input_ids=src_input, attention_mask=src_mask, do_sample=True,
                                            max_length=self.max_decode_len, top_p=top_p,
                                            num_return_sequences=return_seq, temperature=temperature,
                                            num_beams=num_beams)
        sample_input = sample_output[:, :-1].contiguous()
        sample_labels = sample_output[:, 1:].contiguous()
        bsz, sample_len = sample_input.size()

        # keep track of decoded result
        decoded_result_list = []
        for predicted_ids in sample_labels:
            one_result = self.tokenizer.decode(predicted_ids, skip_special_tokens=True)
            decoded_result_list.append(one_result)

        # Repeat source input amd mask to match the sample output shape
        src_input_repeated = src_input.repeat(return_seq, 1)
        src_mask_repeated = src_mask.repeat(return_seq, 1)

        # get sampled loglikelihood
        outputs = self.model(input_ids=src_input_repeated, attention_mask=src_mask_repeated,
                             decoder_input_ids=sample_input,
                             labels=sample_labels, return_dict=True)

        logits = outputs[1]
        assert logits.size() == torch.Size([bsz, sample_len, self.vocab_size])
        logprobs = self.logsftmax(logits)
        unsequeeze_sample_labels = sample_labels.unsqueeze(-1)
        gathered_logprobs = torch.gather(logprobs, dim=-1, index=unsequeeze_sample_labels).squeeze(-1)
        gathered_logprobs = gathered_logprobs.masked_fill(sample_labels.eq(self.padding_idx), float(0.0))

        indicator_matrix = torch.ones_like(gathered_logprobs).type(gathered_logprobs.type())
        indicator_matrix = indicator_matrix.masked_fill(sample_labels.eq(self.padding_idx), float(0.0))

        return gathered_logprobs, indicator_matrix, decoded_result_list
