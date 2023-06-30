import pyhocon
import torch
import numpy as np
import random
from transformers import T5Tokenizer, BartTokenizer


def initialize_config(config_name):
    """Intitalise the config parameters

    Args:
        config_name ([string]): [path to config file]

    Returns:
        [dict]: [config in dict format]
    """

    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[config_name]
    return config


def deterministic_behaviour():
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2 ** 32 - 1)
    np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2 ** 32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2 ** 32 - 1)


def initialise_tokenizer(model_path: str):
    print(f"Loading pre-trained tokenizer from: {model_path}")

    if "t5" in model_path:
        return T5Tokenizer.from_pretrained(model_path)
    elif "bart" in model_path:
        return BartTokenizer.from_pretrained(model_path)
