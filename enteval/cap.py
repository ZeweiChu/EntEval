# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
SNLI - Entailment
'''
from __future__ import absolute_import, division, unicode_literals

import codecs
import os
import io
import copy
import logging
import numpy as np

from enteval.tools.validation import SplitClassifier


class CAPEval(object):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : Coreference Arc Prediction binary Classification*****')
        self.seed = seed
        logging.debug('***** Task path: {}*****\n\n'.format(taskpath))
        train = self.loadFile(os.path.join(taskpath, 'train.txt'))
        valid = self.loadFile(os.path.join(taskpath, 'dev.txt'))
        test = self.loadFile(os.path.join(taskpath, 'test.txt'))

        
        self.samples = [item[0][0] for item in train] + \
                        [item[1][0] for item in train] + \
                        [item[0][0] for item in valid] + \
                        [item[1][0] for item in valid] + \
                        [item[0][0] for item in test] + \
                        [item[1][0] for item in test]
        self.data = {'train': train,
                     'valid': valid,
                     'test': test}

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def loadFile(self, fpath):
        data = []
        item = []
        i = 1
        with codecs.open(fpath, 'rb', 'utf-8') as fin:
            for line in fin:
                if line.strip() == "":
                    data.append(item)
                    item = []
                    i = 0
                else:
                    if i < 3:
                        # print(line)
                        sentence, start, end = line.strip().split("\t")
                        start = int(start)
                        end = int(end)
                        words = sentence.split()
                        item.append([words, start, end])
                    else:
                        item.append(int(line.strip()))
                i += 1
        return data


    def run(self, params, batcher):
        self.X, self.y = {}, {}
        for key in self.data:
            if key not in self.X:
                self.X[key] = []
            if key not in self.y:
                self.y[key] = []

            input1 = [item[0] + [None] for item in self.data[key]]
            input2 = [item[1] + [None] for item in self.data[key]]
            labels = np.array([item[2] for item in self.data[key]])
            
            enc_input = []
            n_labels = len(labels)
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
            self.y[key] = labels

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
        logging.debug('Dev acc : {0} Test acc : {1} for PreCo\n'
                      .format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.data['valid']),
                'ntest': len(self.data['test'])}
