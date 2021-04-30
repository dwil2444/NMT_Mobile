import random
import re
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import unicodedata

from nmt_model import decoder, encoder

import nltk
from nltk.translate.bleu_score import corpus_bleu

def main():
    hidden_size = 256

    encoder1 = encoder.EncoderRNN(10735, hidden_size)
    attn_decoder1 = decoder.AttnDecoderRNN(hidden_size, 17971, drop_rate=0.1)

    encoder1.load_state_dict(torch.load('encoder.pth'))
    attn_decoder1.load_state_dict(torch.load('decoder.pth'))

    print(encoder1)

if __name__ == "__main__":
    main()