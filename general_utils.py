"""Constants and functions of general use."""

import torch
import numpy as np
import random


SEQ_TYPES = ['SRC', 'TGT']
PAD_TOKEN = '<PAD>'
BOS_TOKEN = '<BOS>'
EOS_TOKEN = '<EOS>'
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2


def init_random_seed(value=42):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
