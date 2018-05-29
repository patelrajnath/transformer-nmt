#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 Copyright (c) 2018 Raj Nath Patel
 Licensed under the GNU Public License.
 Author: Raj Nath Patel
 Email: patelrajnath (at) gmail (dot) com
 Created: 26/May/2018 01:04
 """
from __future__ import print_function
import torch
import os

from transformer.utils.data import MyIterator, rebatch
from transformer.models.transformer import make_model
from transformer.utils.download import get_data
from transformer.optim.regularization import LabelSmoothing
from transformer.nmtutils.utils_training import batch_size_fn, run_epoch, SimpleLossCompute, \
    save_state, load_model_state
from transformer.optim.noam import NoamOpt
from transformer.translate.decode_transformer import greedy_decode


# GPUs to use
is_cuda=False
train, val, test, SRC, TGT = get_data()

if True:
    pad_idx = TGT.vocab.stoi["<blank>"]
    global model
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)

    if is_cuda:
        cuda_device = os.environ["CUDA_VISIBLE_DEVICES"]
        print(cuda_device)
        model.cuda()
        criterion.cuda()
    else:
        cuda_device = -1

    BATCH_SIZE = 1000
    global train_iter
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=cuda_device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    global valid_iter
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=cuda_device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    # model_par = nn.DataParallel(model, device_ids=devices)
None

checkpoint = 'filename.pt'

run_training=True
if run_training:
    global model_opt
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    start_epoch = 0
    max_epochs = 10
    start_epoch = load_model_state(checkpoint, model, cuda_device=cuda_device)
    for epoch in range(start_epoch, max_epochs):
        model.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter),
                  model,
                  SimpleLossCompute(model.generator, criterion, model_opt), epoch)
        save_state(checkpoint, model, criterion, model_opt, epoch)
        model.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model,
                         SimpleLossCompute(model.generator, criterion, None))
        print(loss)
else:
    start_epoch = load_model_state(checkpoint, model, cuda_device=cuda_device)
    model.eval()

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
