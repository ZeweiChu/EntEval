# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import logging
import torch
from collections import OrderedDict
import pickle
import os
import code

# Set PATHs
PATH_TO_ENTEVAL = '..'
PATH_TO_DATA = '../data'

ELMO_PATH="entelmo"
sys.path.insert(0, ELMO_PATH)
from elmo import Elmo, batch_to_ids
# Dump: elmo_dump.hdf5
# Options: options.json
# Code: elmo.py
options_file = os.path.join(MINGDA_ELMO_PATH, "options.json")
weight_file = os.path.join(MINGDA_ELMO_PATH, "elmo_dump.hdf5")


# import EntEval
sys.path.insert(0, PATH_TO_ENTEVAL)
import enteval

def prepare(params, samples):
    return

def batcher(params, batch):
    use_ctx = False
    use_def = False
    if batch[0][0] is not None:
        use_ctx = True
    if batch[0][3] is not None:    
        use_def = True

    if use_ctx: 
        max_context_len = max([len(item[0]) for item in batch])
        batch_contexts = [] 
        batch_context_span_mask = np.zeros((len(batch), max_context_len)).astype("float32")
    if use_def:    
        batch_desc = []
    
    for i, (ctx, s, e, desc) in enumerate(batch):
        if use_ctx:
            batch_contexts.append([w.lower() for w in ctx] if ctx != [] else ['.'])
            batch_context_span_mask[i, s:] = 1.
            batch_context_span_mask[i, e:] = 0.
        if use_def:
            batch_desc.append([w.lower() for w in desc] if desc != [] else ['.'])

    context_embedding = None
    def_embedding = None
    with torch.no_grad():
        if use_ctx:
            char_ids = batch_to_ids(batch_contexts).cuda().contiguous()
            elmo_outputs = elmo(char_ids, if_context=True)
            embeddings = elmo_outputs["elmo_representations"][0]
            mask = torch.from_numpy(batch_context_span_mask).cuda().unsqueeze(-1)
            embeddings = torch.sum(embeddings * mask, 1) / torch.sum(mask, 1)
            context_embedding = embeddings.cpu().data.numpy()
        if use_def:
            char_ids = batch_to_ids(batch_desc).cuda().contiguous()
            elmo_outputs = elmo(char_ids, if_context=False)
            embeddings = elmo_outputs["elmo_representations"][0]
            mask = elmo_outputs["mask"].unsqueeze(-1).float()
            embeddings = torch.sum(embeddings * mask, 1) / torch.sum(mask, 1)
            def_embedding = embeddings.data.cpu().numpy()

    return context_embedding, def_embedding


# Set params for EntEval
params_enteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'batch_size': 8}
params_enteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    layer = sys.argv[1]
    task = int(sys.argv[2])
    mix_params = np.ones(3)
    if layer in ['0', '1', '2']:
        layer = int(layer)
        mix_params *= -1e4
        mix_params[layer] = 1e4 
    
    elmo = Elmo(options_file, weight_file, 1, dropout=0, scalar_mix_parameters=mix_params).cuda()

    print("Using {} layer of ElMo".format(layer))

    se = enteval.engine.SE(params_enteval, batcher, prepare)
    task_groups = [
        ["CAPsame", "CAPnext", "CERP", "EFP", "KORE", "WikiSRS", "ERT"],
        ["Rare"],
        ["ET"],
        ["ConllYago"],
    ]

    results = se.eval(task_groups[task])
    for k, v in results.items():
        print(k, v)

