import math
import os
import sys

import kaldi_io
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from torch_dataset import TorchSpeechDataset
from torch_dnn import TorchDNN

if len(sys.argv) < 3:
    print("USAGE: python extract_posteriors.py <MY_TORCHDNN_CHECKPOINT> <OUTPUT_DIR>")

CHECKPOINT_TO_LOAD = '/home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/checkpoint/weights.pt'
OUT_DIR = '/home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/checkpoint/output'

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
OUTPUT_ARK_FILE = os.path.join(OUT_DIR, "posteriors.ark")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# FIXME: You may need to change these paths
TRAIN_ALIGNMENT_DIR = '/home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/triphone_train_ali'
TEST_ALIGNMENT_DIR = '/home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/triphone_test_ali'


def extract_logits(model, test_loader):
    """Runs through the  test_loader and returns a
    tensor containing the logits (forward output) for each sample in the test set
    """
    model.eval()
    outs = []
    for i,data in enumerate(test_loader):
        x, y = data 
        batch_out = model(x)
        outs.append(batch_out)

    # print(outs.size())
    # print(outs[0].size())
    print("logits ready to concat")
    output = outs[0]
    print(len(outs))
    i=0
    for tensor in outs[1:] : 
        print(i)
        i+=1
        # print(tensor.size())
        out = torch.cat((output, tensor), dim=0)
        output=out
        # print(out.size())
        
    print(output.size())
    return output


trainset = TorchSpeechDataset('./', TRAIN_ALIGNMENT_DIR, 'train')
testset = TorchSpeechDataset('./', TEST_ALIGNMENT_DIR, 'test')


scaler = StandardScaler()
scaler.fit(trainset.feats)

testset.feats = scaler.transform(testset.feats)

test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)


dnn = torch.load(CHECKPOINT_TO_LOAD, map_location="cpu").to(DEVICE)

print("start computing logits...")
logits = extract_logits(dnn, test_loader)


post_file = kaldi_io.open_or_fd(OUTPUT_ARK_FILE, 'wb')

start_index = 0
testset.end_indices[-1] += 1

for i, name in enumerate(testset.uttids):
    out = logits[start_index:testset.end_indices[i]].detach().cpu().numpy()
    start_index = testset.end_indices[i]
    kaldi_io.write_mat(post_file, out, testset.uttids[i])
