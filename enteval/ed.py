# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
Entity Disembiguation task
'''
from __future__ import absolute_import, division, unicode_literals

import codecs
import os
import io
import copy
import logging
import numpy as np
import code

from enteval.tools.validation import SplitMultiClassClassifier
from enteval.tools.validation import SplitClassifier


class RareEval(object):
    def __init__(self, taskpath, use_name=False, seed=1111):
        logging.debug('***** Transfer task : Entity Disembiguation prediction *****\n\n')
        self.seed = seed
        self.use_name = use_name
        if self.use_name:
            logging.debug("***** Use entity names to compute embedding *****")
        else:
            logging.debug("***** Use entity descriptions to compute embedding *****")

        train_labels, train_context, train_desc = self.loadFile(os.path.join(taskpath, 'train.txt'))
        valid_labels, valid_context, valid_desc = self.loadFile(os.path.join(taskpath, 'valid.txt'))
        test_labels, test_context, test_desc = self.loadFile(os.path.join(taskpath, 'test.txt'))
        self.data = {'train': (train_labels, train_context, train_desc),
                     'valid': (valid_labels, valid_context, valid_desc),
                     'test': (test_labels, test_context, test_desc)}
        self.samples = [sent[0] for sents in train_context for sent in sents] + \
                        [sent for sents in train_desc for sent in sents] + \
                        [sent[0] for sents in valid_context for sent in sents] + \
                        [sent for sents in valid_desc for sent in sents] + \
                        [sent[0] for sents in test_context for sent in sents] + \
                        [sent for sents in test_desc for sent in sents] 

        # code.interact(local=locals())

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def process_desc(self, desc):
        sents = desc.split("\t")
        for sent in sents:
            if "**blank**" in sent:
                words = sent.split()
                break

        index = words.index("**blank**")
        words = words[max(0, index-150):index+150]
        index = words.index("**blank**")
        return words, index

    def loadFile(self, fpath):
        labels, contexts, descs = [], [], []
        data = []
        with open(fpath, 'r') as f:
            for line in f:
                if line.strip() == "":
                    # code.interact(local=locals())
                    contexts.append([])
                    descs.append([])
                    words, index = self.process_desc(data[0])
                    for context in data[1:5]:
                        # print(context)
                        entity, desc = context.split("\t")
                        entity = entity.split()
                        desc = desc.split()
                        new_words = words[:index] + entity + words[index+1:]
                        contexts[-1].append([new_words, index, index+len(entity)])
                        if self.use_name:
                            descs[-1].append(entity)
                        else:
                            descs[-1].append(desc)
                    labels.append(int(data[5]))
                    data = []
                else:
                    data.append(line.strip())
        return labels, contexts, descs

    def run(self, params, batcher):
        self.X, self.y = {}, {}
        for key in self.data:
            if key not in self.X:
                self.X[key] = []
            if key not in self.y:
                self.y[key] = []

            labels, contexts, descs = self.data[key]
            enc_input = []
            n_labels = len(labels)
            for ii in range(0, n_labels, params.batch_size):
                batch_context = [b for batch in contexts[ii:ii + params.batch_size] for b in batch]
                batch_desc = [b for batch in descs[ii:ii + params.batch_size] for b in batch]
                batch = [a + [b] for a,b in zip(batch_context, batch_desc)]
                if len(batch) > 0: #== params.batch_size:
                    context_enc, desc_enc = batcher(params, batch)
                    # code.interact(local=locals())
                    enc_input.append(np.hstack((context_enc, desc_enc, context_enc * desc_enc, np.abs(context_enc - desc_enc))))
                if (ii) % (20000*params.batch_size) == 0:
                    logging.info("PROGRESS (encoding): %.2f%%" %
                                 (100 * ii / n_labels))
            self.X[key] = np.vstack(enc_input)
            self.X[key] = self.X[key].reshape(-1, 4, self.X[key].shape[-1])
            self.y[key] = np.array(labels) #[:len(self.X[key])])
            # code.interact(local=locals())
        logging.debug("Training data shape: {}".format(self.X["train"].shape))
        config = {'nclasses': 4, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'cudaEfficient': True,
                  'nhid': 2000, 'noreg': False}

        config_classifier = copy.deepcopy(params.classifier)
        config_classifier['max_epoch'] = 15
        config_classifier['epoch_size'] = 1
        config_classifier['nhid'] = 0#2000
        config['classifier'] = config_classifier

        clf = SplitMultiClassClassifier(self.X, self.y, config)
        devacc, testacc = clf.run()
        logging.debug('Dev acc : {0} Test acc : {1} for Rare Entity Prediction\n'
                      .format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.data['valid'][0]),
                'ntest': len(self.data['test'][0])}




class ConllYagoEval(object):
    def __init__(self, taskpath, use_name=False, seed=1111):
        logging.debug('***** Transfer task : Conll Yago Entity Linking *****\n\n')
        self.seed = seed
        self.use_name = use_name
        if self.use_name:
            logging.debug("***** Use entity names to compute embedding *****")
        else:
            logging.debug("***** Use entity descriptions to compute embedding *****")

        train_context, train_desc = self.loadFile(os.path.join(taskpath, 'train.final.txt'))
        valid_context, valid_desc = self.loadFile(os.path.join(taskpath, 'testa.final.txt'))
        test_context, test_desc = self.loadFile(os.path.join(taskpath, 'testb.final.txt'))
        self.data = {'train': (train_context, train_desc),
                     'valid': (valid_context, valid_desc),
                     'test': (test_context, test_desc)}
        self.samples = [ins[0] for ins in train_context] + \
                        [ins[0] for ins in valid_context] + \
                        [ins[0] for ins in test_context] + \
                        [cand[0] for cands in train_desc for cand in cands] + \
                        [cand[0] for cands in valid_desc for cand in cands] + \
                        [cand[0] for cands in test_desc for cand in cands]

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def loadFile(self, fpath):
        contexts, descs = [], []
        with open(fpath, 'r') as f:
            for line in f:
                s, e, sent, _, entities = line.strip().split("\t", 4)
                entities = entities.split("\t")
                # code.interact(local=locals())
                _descs = []
                for entity in entities:
                    # print(entity)
                    prior, entity_name, desc = entity.split("|||", 2)
                    if self.use_name:
                        entity_name = entity_name.split("_")
                        _descs.append([entity_name, float(prior), 0])
                    else:
                        desc = desc.split()
                        _descs.append([desc, float(prior), 0])
                _descs[0][-1] = 1
                
                contexts.append([sent.split(), int(s), int(e)])
                descs.append(_descs)
                
        # context: [[sent, s, e] * N]
        # descs: [[[desc, prior, 0/1] * M] * N]
        return contexts, descs

    def run(self, params, batcher):
        self.X, self.y, self.priors = {}, {}, {}
        for key in self.data:
            if key not in self.X:
                self.X[key] = []
            if key not in self.y:
                self.y[key] = []

            contexts, descs = self.data[key]
            n_contexts = len(contexts)
            enc_contexts = []
            for ii in range(0, n_contexts, params.batch_size):
                batch_context = [item + [None] for item in contexts[ii:ii + params.batch_size]]
                # code.interact(local=locals())
                enc_context, _ = batcher(params, batch_context)
                enc_contexts.append(enc_context)
            enc_contexts = np.vstack(enc_contexts)

            enc_descs = []
            labels = [desc[-1] for _descs in descs for desc in _descs]
            priors = [desc[-2] for _descs in descs for desc in _descs]
            num_descs = [len(_descs) for _descs in descs]
            descs = [desc[0] for _descs in descs for desc in _descs]
            n_descs = len(descs)
            for ii in range(0, n_descs, params.batch_size):
                batch_descs = [[None, None, None] + [item] for item in descs[ii:ii + params.batch_size]]

                _, enc_desc = batcher(params, batch_descs)
                enc_descs.append(enc_desc)
            enc_descs = np.vstack(enc_descs)

            enc_contexts = [enc_contexts[i] for i in range(len(num_descs)) for j in range(num_descs[i])]
            enc_contexts = np.vstack(enc_contexts)
            labels = np.array(labels).astype("int64")
            priors = np.array(priors).astype("float32")
            print(enc_contexts.shape, enc_descs.shape)
#            self.X[key]["context"] = enc_contexts.reshape(enc_descs.shape)
#            self.X[key]["desc"] = enc_descs

            self.X[key] = np.concatenate([enc_contexts, enc_descs, enc_contexts*enc_descs, np.abs(enc_contexts-enc_descs)], 1)
            self.y[key] = labels
            self.priors[key] = priors

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'cudaEfficient': True,
                  'noreg': False}

        config_classifier = copy.deepcopy(params.classifier)
        config_classifier['max_epoch'] = 15
        config_classifier['epoch_size'] = 1
        config['classifier'] = config_classifier

        clf = SplitClassifier(self.X, self.y, config)
        devacc, testacc = clf.run(return_score=True)
        testacc, all_logits = testacc
        _, descs = self.data[key]
        num_descs = [len(_descs) for _descs in descs]
        cums = [0] + np.cumsum(num_descs).tolist()
        all_logits_v1 = all_logits + self.priors["test"] - all_logits * self.priors["test"]
        # code.interact(local=locals()) 
        preds_v1 = np.array([np.argmax(all_logits_v1[cums[i]:cums[i+1]]) for i in range(len(cums)-1)])
        testacc_v1 = (preds_v1==0).sum()/len(preds_v1)
        all_logits_v2 = all_logits + self.priors["test"]
        preds_v2 = np.array([np.argmax(all_logits_v2[cums[i]:cums[i+1]]) for i in range(len(cums)-1)])
        testacc_v2 = (preds_v2==0).sum()/len(preds_v2)
        all_logits_v3 = all_logits * self.priors["test"]
        preds_v3 = np.array([np.argmax(all_logits_v3[cums[i]:cums[i+1]]) for i in range(len(cums)-1)])
        testacc_v3 = (preds_v3==0).sum()/len(preds_v3)
        all_logits_prior = self.priors["test"]
        preds_prior = np.array([np.argmax(all_logits_prior[cums[i]:cums[i+1]]) for i in range(len(cums)-1)])
        testacc_prior = (preds_prior==0).sum()/len(preds_prior)

        logging.debug('Dev acc : {0} Test acc v1: {1} Test acc v2: {2} Test acc v3: {3} Test acc prior: {4} for ConNLL Yago Entity Linking\n'
                      .format(devacc, testacc_v1, testacc_v2, testacc_v3, testacc_prior))
        return {'devacc': devacc, 'binarytestacc': testacc, 'testaccv1': testacc_v1,
                'testaccv2': testacc_v2, 'testaccv3': testacc_v3, 'testacc_prior': testacc_prior, 
                'ndev': len(self.data['valid'][0]),
                'ntest': len(self.data['test'][0])}
