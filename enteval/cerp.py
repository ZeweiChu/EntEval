# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
Contexualized Entity Relationship Prediction
'''
from __future__ import absolute_import, division, unicode_literals

import codecs
import os
import io
import copy
import logging
import numpy as np
import code

from enteval.tools.validation import SplitClassifier


class CERPEval(object):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : Contexualized Entity Relationship Prediction *****\n\n')
        self.seed = seed
        trainlabels, train1, train2 = self.loadFile(os.path.join(taskpath, 'train.txt'))
        validlabels, valid1, valid2 = self.loadFile(os.path.join(taskpath, 'dev.txt'))
        testlabels, test1, test2 = self.loadFile(os.path.join(taskpath, 'test.txt'))

        self.samples = [data[0] for data in train1] + [data[0] for data in train2] + [data[0] for data in valid1] + [data[0] for data in valid2] + [data[0] for data in test1] + [data[0] for data in test2]
        self.data = {'train': (train1, train2, trainlabels),
                     'valid': (valid1, valid2, validlabels),
                     'test': (test1, test2, testlabels)
                     }

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def loadFile(self, fpath):
        labels, entities1, entities2 = [], [], []
        with codecs.open(fpath, 'rb', 'utf-8') as f:
            for line in f:
                
                label, s1, e1, s2, e2, sentence = line.strip().split("\t")
                sentence = sentence.split()
                # print(line)
                labels.append(int(label))
                entities1.append([sentence, int(s1), int(e1), None])
                entities2.append([sentence, int(s2), int(e2), None])
                
        return labels, entities1, entities2
            

    def run(self, params, batcher):
        self.X, self.y = {}, {}
        for key in self.data:
            if key not in self.X:
                self.X[key] = []
            if key not in self.y:
                self.y[key] = []

            input1, input2, mylabels = self.data[key]
            enc_input = []
            n_labels = len(mylabels)
            for ii in range(0, n_labels, params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]

                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1, _ = batcher(params, batch1)
                    enc2, _ = batcher(params, batch2)
                    enc_input.append(np.hstack((enc1, enc2, enc1 * enc2,
                                                np.abs(enc1 - enc2))))
                if (ii*params.batch_size) % (20000*params.batch_size) == 0:
                    logging.info("PROGRESS (encoding): %.2f%%" %
                                 (100 * ii / n_labels))
            self.X[key] = np.vstack(enc_input)
            self.y[key] = np.array(mylabels)

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'cudaEfficient': True,
                  'nhid': params.nhid, 'noreg': True}

        config_classifier = copy.deepcopy(params.classifier)
        config_classifier['max_epoch'] = 15
        config_classifier['epoch_size'] = 1
        config['classifier'] = config_classifier

        clf = SplitClassifier(self.X, self.y, config)
        devacc, testacc = clf.run()
        logging.debug('Dev acc : {0} Test acc : {1} for Contexualized Entity Relationship Prediction \n'
                      .format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.data['valid'][0]),
                'ntest': len(self.data['test'][0])}
