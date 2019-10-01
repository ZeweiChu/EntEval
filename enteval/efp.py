# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
Entity Factuality Prediction classification
'''
from __future__ import absolute_import, division, unicode_literals

import io
import os
import numpy as np
import logging

from enteval.tools.validation import SplitClassifier


class EFPEval(object):
    def __init__(self, task_path, use_ctx=True, nclasses=2, seed=1111):
        logging.debug('***** Transfer task : Entity Factuality Prediction classification *****\n\n')
        self.nclasses = nclasses
        self.task_name = "Fever"
        self.seed = seed
        self.use_ctx = use_ctx

        trainlabels, traintext = self.loadFile(os.path.join(task_path, 'train.txt'))
        devlabels, devtext = self.loadFile(os.path.join(task_path, 'dev.txt'))
        testlabels, testtext = self.loadFile(os.path.join(task_path, 'test.txt'))
        self.id2label = list(set(trainlabels))
        self.label2id = {label:i for i, label in enumerate(self.id2label)}
        trainlabels = [self.label2id[l] for l in trainlabels]
        devlabels = [self.label2id[l] for l in devlabels]
        testlabels = [self.label2id[l] for l in testlabels]
        self.data = {'train': [trainlabels, traintext], 'dev': [devlabels, devtext], 'test': [testlabels, testtext]}

        if self.use_ctx:
            self.samples = [item[0] for item in traintext] + \
                        [item[0] for item in devtext] + \
                        [item[0] for item in testtext]
        else:
            self.samples = [item[-1] for item in traintext] + \
                        [item[-1] for item in devtext] + \
                        [item[-1] for item in testtext]

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def loadFile(self, fpath):
        labels = []
        data = []
        with io.open(fpath, 'r', encoding='utf-8') as fin:
            for line in fin:
                label, entity, s, e = line.strip().split("\t")
                labels.append(label)
                if self.use_ctx:
                    data.append([entity.split(), int(s), int(e), None])
                else:
                    data.append([None, None, None, entity.split()])
        return labels, data


    def run(self, params, batcher):
        fever_embed = {'train': {}, 'dev': {}, 'test': {}}
        bsize = params.batch_size

        for key in self.data:
            logging.info('Computing embedding for {0}'.format(key))
            
            fever_embed[key]['X'] = []
            for ii in range(0, len(self.data[key][1]), bsize):
                batch = self.data[key][1][ii:ii + bsize]
                if self.use_ctx:
                    embeddings, _ = batcher(params, batch)
                else:
                    _, embeddings = batcher(params, batch)
                fever_embed[key]['X'].append(embeddings)
            fever_embed[key]['X'] = np.vstack(fever_embed[key]['X'])
            fever_embed[key]['y'] = np.array(self.data[key][0])
            logging.info('Computed {0} embeddings'.format(key))

        config_classifier = {'nclasses': self.nclasses, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier}

        clf = SplitClassifier(X={'train': fever_embed['train']['X'],
                                 'valid': fever_embed['dev']['X'],
                                 'test': fever_embed['test']['X']},
                              y={'train': fever_embed['train']['y'],
                                 'valid': fever_embed['dev']['y'],
                                 'test': fever_embed['test']['y']},
                              config=config_classifier)

        devacc, testacc = clf.run()
        logging.debug('\nDev acc : {0} Test acc : {1} for \
            {2} classification with {3} encoder \n'.format(devacc, testacc, self.task_name, "context" if self.use_ctx else "description"))

        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(fever_embed['dev']['X']),
                'ntest': len(fever_embed['test']['X'])}
