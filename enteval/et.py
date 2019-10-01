# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
entity typing classification
'''
from __future__ import absolute_import, division, unicode_literals

import io
import os
import numpy as np
import random
import logging
import torch
import torch.nn as nn
import copy

from sklearn.metrics import precision_score, recall_score, f1_score

class ETEval(object):
    def __init__(self, task_path, nclasses=2, seed=1111):
        logging.debug('***** Transfer task : Entity Typing classification *****\n\n')
        self.nclasses = nclasses
        self.task_name = "ET"
        self.seed = seed
        self.task_path = task_path

        self.load_labels()
        traindata = self.loadFile(os.path.join(task_path, 'train.txt'))
        devdata = self.loadFile(os.path.join(task_path, 'valid.txt'))
        testdata = self.loadFile(os.path.join(task_path, 'test.txt'))
        
        self.data = {'train': traindata, 'dev': devdata, 'test': testdata}
        self.samples = [item[0] for item in traindata] + \
                        [item[0] for item in devdata] + \
                        [item[0] for item in testdata]

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def load_labels(self):
        self.id2label = []
        with open(os.path.join(self.task_path, 'labels.txt')) as fin:
            for line in fin:
                self.id2label.append(line.strip())
        self.label2id = {label:i for i, label in enumerate(self.id2label)}


    def loadFile(self, fpath):
        data = []
        
        with io.open(fpath, 'r', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip().split("\t")
                entity = line[0].split()
                s = int(line[1])
                e = int(line[2])
                pos_labels = line[3:]

                data.append([entity, s, e, None, [self.label2id[label] for label in pos_labels], 1])

                neg_labels = list(set(self.id2label).difference(set(pos_labels)))

                data.append([entity, s, e, None, [self.label2id[label] for label in neg_labels], 0])

        return data

    def run(self, params, batcher):
        ultra_embed = {'train': {}, 'dev': {}, 'test': {}}
        bsize = params.batch_size

        for key in self.data:
            logging.info('Computing embedding for {0}'.format(key))
            
            ultra_embed[key]['X'] = []
            for ii in range(0, len(self.data[key]), bsize):
                batch = [d[:4] for d in self.data[key][ii:ii + bsize]]
                embeddings, _ = batcher(params, batch)
                ultra_embed[key]['X'].append(embeddings)

            ultra_embed[key]['X'] = np.vstack(ultra_embed[key]['X'])
            X = []
            labels = []
            ys = []
            # repeat len(label) times
            for i, (entity, s, e, _, label, y) in enumerate(self.data[key]):
                X += [ultra_embed[key]['X'][i]] * len(label)
                labels += label
                ys += [y] * len(label)
            
            ultra_embed[key]['X'] = np.vstack(X).astype("float32")
            ultra_embed[key]['label'] = np.array(labels).astype("int64")
            ultra_embed[key]['y'] = np.array(ys).astype("float32")
            # code.interact(local=locals())
            logging.info('Computed {0} embeddings'.format(key))

        # code.interact(local=locals())
        input_dim = ultra_embed['train']['X'].shape[1]
        clf = MLPClassifier(input_dim=input_dim,
                            label2id=self.label2id,
                            X={'train': ultra_embed['train']['X'],
                            'dev': ultra_embed['dev']['X'],
                            'test': ultra_embed['test']['X']},
                            label={'train': ultra_embed['train']['label'],
                            'dev': ultra_embed['dev']['label'],
                            'test': ultra_embed['test']['label']},
                            y={'train': ultra_embed['train']['y'],
                            'dev': ultra_embed['dev']['y'],
                            'test': ultra_embed['test']['y']})

        dev_f1, testacc, prec, recall, f1 = clf.run()
        logging.debug('\nDev f1 : {} Test acc : {} Precision: {} Recall: {} F1 score: {} '
            '{} classification\n'.format(dev_f1, testacc, prec, recall, f1, self.task_name))

        return {'dev_f1': dev_f1, 'acc': testacc, 'precision': prec, 
                "recall": recall, "f1": f1, 
                'ndev': len(ultra_embed['dev']['X']),
                'ntest': len(ultra_embed['test']['X'])}


class Model(nn.Module):
    def __init__(self, input_dim, label2id):
        super(Model, self).__init__()
        self.label2id = label2id
        self.labelembed = nn.Embedding(len(label2id), input_dim)

    def forward(self, x, label):
        embeddings = self.labelembed(label)
        scores = torch.sum(x*embeddings, 1)
        return scores

class MLPClassifier():
    def __init__(self, input_dim, label2id, X, label, y):
        self.trainX = X["train"]
        self.trainlabel = label["train"]
        self.trainy = y["train"]
        self.devX = X["dev"]
        self.devlabel = label["dev"]
        self.devy = y["dev"]
        self.testX = X["test"]
        self.testlabel = label["test"]
        self.testy = y["test"]

        self.max_epoch = 15
        self.epoch_size = 4
        self.tenacity = 5
        self.batch_size = 64

        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

        self.model = Model(input_dim, label2id).to(self.device)
        
        self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)
        self.loss_fn.size_average = False
        # self.model.labelembed.weight.requires_grad = False
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), weight_decay=1e-9)

    def fit(self, trainX, trainlabel, trainy, early_stop=True):
        best_f1 = -1.
        early_stop_count = 0
        stop_train = False
        self.nepoch = 0
        while not stop_train and self.nepoch <= self.max_epoch:
            self.trainepoch(trainX, trainlabel, trainy, epoch_size=self.epoch_size)
            acc, prec, recall, f1 = self.evaluate(self.devX, self.devlabel, self.devy, istest=True)
            logging.debug("epoch {} accuracy {} precition {} recall {} f1 {}\n".format(self.nepoch, acc, prec, recall, f1))
            if f1 > best_f1:
                best_f1 = f1
                bestmodel = copy.deepcopy(self.model)
            elif early_stop:
                if early_stop_count >= self.tenacity:
                    stop_train = True
                early_stop_count += 1
        self.model = bestmodel
        return best_f1

    def evaluate(self, devX, devlabels, devy, threshold=0.5, istest=False):
        self.model.eval()
        correct = 0
        all_outputs = []
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                labelbatch = devlabels[i:i+self.batch_size]
                ybatch = devy[i:i + self.batch_size]

                Xbatch = torch.from_numpy(Xbatch).to(self.device)
                labelbatch = torch.from_numpy(labelbatch).to(self.device)
                ybatch = torch.from_numpy(ybatch).to(self.device)

                output = torch.sigmoid(self.model(Xbatch, labelbatch))
                # code.interact(local=locals())
                output[output >= threshold] = 1
                output[output < threshold] = 0
                correct += output.long().eq(ybatch.data.long()).sum().item()
                all_outputs.append(output.data.cpu().numpy())
            accuracy = 1.0 * correct / devX.shape[0]

        if istest:
            all_outputs = np.concatenate(all_outputs, 0)
            prec = precision_score(devy, all_outputs)
            recall = recall_score(devy, all_outputs)
            f1 = f1_score(devy, all_outputs)

            return accuracy, prec, recall, f1
        else:
            return accuracy

    def trainepoch(self, X, labels, y, epoch_size=1):
        self.model.train()
        for _ in range(self.nepoch, self.nepoch + epoch_size):
            permutation = np.random.permutation(len(X)).astype("int64")
            all_costs = []
            for i in range(0, len(X), self.batch_size):
                # forward
                idx = permutation[i:i + self.batch_size]
                Xbatch = torch.from_numpy(X[idx]).to(self.device)
                labelbatch = torch.from_numpy(labels[idx]).to(self.device)
                ybatch = torch.from_numpy(y[idx]).to(self.device)

                output = self.model(Xbatch, labelbatch)
                # loss
                loss = self.loss_fn(output, ybatch)
                all_costs.append(loss.data.item())
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                # Update parameters
                self.optimizer.step()
        self.nepoch += epoch_size

    def run(self):
        best_model = self.fit(self.trainX, self.trainlabel, self.trainy)
        best_dev_f1 = -1
        best_threshold = 0.
        logging.debug("Tuning the optimal threshold on the dev set")
        num_to_tune = 20
        for i in range(num_to_tune + 1):
            threshold = i / num_to_tune
            acc, prec, recall, f1 = self.evaluate(self.devX, self.devlabel, self.devy, threshold=threshold, istest=True)
            logging.debug("threshold {} accuracy {} precition {} recall {} f1 {}\n".format(threshold, acc, prec, recall, f1))
            if f1 > best_dev_f1:
                best_dev_f1 = f1
                best_threshold = threshold
        logging.debug("best threshold: {}".format(best_threshold))
        acc, prec, recall, f1 = self.evaluate(self.testX, self.testlabel, self.testy, best_threshold, istest=True)
        return best_dev_f1, acc, prec, recall, f1


