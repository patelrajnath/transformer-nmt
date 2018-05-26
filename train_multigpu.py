#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
 Copyright (c) 2018 Raj Nath Patel
 Licensed under the GNU Public License.
 Author: Raj Nath Patel
 Email: patelrajnath (at) gmail (dot) com
 Created: 26/May/2018 01:04
 """
from __future__ import print_function

import os

import torch

from utils.data import MyIterator, rebatch
from models.transformer import make_model
from utils.download import get_data
from optim.regularization import LabelSmoothing
from nmtutils.utils_training import batch_size_fn, run_epoch, SimpleLossCompute, save_state
from optim.noam import NoamOpt
from translate.decode_transformer import greedy_decode


# GPUs to use
devices = [0, 1, 2, 3]
train, val, test, SRC, TGT = get_data()

def does_file_exists(filePathAndName):
    return os.path.exists(filePathAndName)

if True:
    pad_idx = TGT.vocab.stoi["<blank>"]
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    model.cuda()

    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()
    BATCH_SIZE = 1000
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=None,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=None,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
    # model_par = nn.DataParallel(model, device_ids=devices)
None

checkpoint = 'filename.pt'

if True:
    if does_file_exists(checkpoint):
        print("Loading model from checkpoints", checkpoint)
        model = torch.load('filename.pt')
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(10):
        model.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter), model,
                  SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch((rebatch(pad_idx, b) for b in train_iter), model,
                        SimpleLossCompute(model.generator, criterion, None)))
        torch.save(model, checkpoint)

else:
    model = torch.load(checkpoint)

for i, batch in enumerate(valid_iter):
    src = batch.src.transpose(0, 1)[:1]
    src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
    out = greedy_decode(model, src, src_mask,
                        max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
    print ("Translation:", end="\t")
    for i in range(1, out.size(1)):
        sym = TGT.vocab.itos[out[0, i]]
        if sym == "</s>": break
        print (sym, end =" ")
    print()
    print ("Target:", end="\t")
    for i in range(1, batch.trg.size(0)):
        sym = TGT.vocab.itos[batch.trg.data[i, 0]]
        if sym == "</s>": break
        print (sym, end =" ")
    print()
    break
