# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
Entity relatedness type prediction
'''
from __future__ import absolute_import, division, unicode_literals

import codecs
import os
import io
import copy
import logging
import numpy as np

from enteval.tools.validation import SplitClassifier


class ERTEval(object):
    def __init__(self, taskpath, use_name=False, seed=1111):
        logging.debug('***** Transfer task : Entity Relationship prediction *****')
        logging.debug('***** task path: {} *****\n\n'.format(taskpath))
        self.seed = seed
        self.use_name = use_name
        if self.use_name:
            logging.debug("***** Use entity names to compute embedding *****")
        else:
            logging.debug("***** Use entity descriptions to compute embedding *****")
        trainlabels, train1, train2 = self.loadFile(os.path.join(taskpath, 'train.txt'))
        validlabels, valid1, valid2 = self.loadFile(os.path.join(taskpath, 'dev.txt'))
        testlabels, test1, test2 = self.loadFile(os.path.join(taskpath, 'test.txt'))

        self.id2label = list(set(trainlabels))
        self.label2id = {label:i for i, label in enumerate(self.id2label)}
        trainlabels = [self.label2id[l] for l in trainlabels]
        validlabels = [self.label2id[l] for l in validlabels]
        testlabels = [self.label2id[l] for l in testlabels]


        # sort data (by s2 first) to reduce padding
        sorted_train = sorted(zip(train2, train1, trainlabels),
                              key=lambda z: (len(z[0]), len(z[1]), z[2]))
        train2, train1, trainlabels = map(list, zip(*sorted_train))

        sorted_valid = sorted(zip(valid2, valid1, validlabels),
                              key=lambda z: (len(z[0]), len(z[1]), z[2]))
        valid2, valid1, validlabels = map(list, zip(*sorted_valid))

        sorted_test = sorted(zip(test2, test1, testlabels),
                             key=lambda z: (len(z[0]), len(z[1]), z[2]))
        test2, test1, testlabels = map(list, zip(*sorted_test))

        self.samples = train1 + train2 + valid1 + valid2 + test1 + test2
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
                label, entity1, entity2, entity_desc1, entity_desc2 = line.strip().split("\t")
                labels.append(label)
                if self.use_name:
                    entities1.append(entity1.split())
                    entities2.append(entity2.split())
                else:
                    entities1.append(entity_desc1.split())
                    entities2.append(entity_desc2.split())
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

                batch1 = [[None, None, None] + [item] for item in batch1]
                batch2 = [[None, None, None] + [item] for item in batch2]

                if len(batch1) == len(batch2) and len(batch1) > 0:
                    _, enc1 = batcher(params, batch1)
                    _, enc2 = batcher(params, batch2)
                    enc_input.append(np.hstack((enc1, enc2, enc1 * enc2,
                                                np.abs(enc1 - enc2))))
                if (ii*params.batch_size) % (20000*params.batch_size) == 0:
                    logging.info("PROGRESS (encoding): %.2f%%" %
                                 (100 * ii / n_labels))
            self.X[key] = np.vstack(enc_input)
            self.y[key] = np.array(mylabels)

        config = {'nclasses': len(self.id2label), 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'cudaEfficient': True,
                  'nhid': params.nhid, 'noreg': True}

        config_classifier = copy.deepcopy(params.classifier)
        config_classifier['max_epoch'] = 15
        config_classifier['epoch_size'] = 1
        config['classifier'] = config_classifier

        clf = SplitClassifier(self.X, self.y, config)
        devacc, testacc = clf.run()
        logging.debug('Dev acc : {0} Test acc : {1} for Entity Relationship\n'
                      .format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.data['valid'][0]),
                'ntest': len(self.data['test'][0])}
