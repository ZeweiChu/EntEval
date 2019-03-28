from __future__ import absolute_import, division, unicode_literals

import os
import io
import numpy as np
import logging

from scipy.stats import spearmanr, pearsonr
from senteval.utils import cosine

class KOREEval(object):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : KORE *****\n\n')
        self.seed = seed
        self.loadFile(taskpath)

    def loadFile(self, fpath):
        self.data = {}
        self.samples = []

        self.head_entities = []
        self.compare_entities = []
        compare = []
        with io.open(fpath + '/all.txt', encoding='utf8') as fin:
            for line in fin:
                start, entity, desc = line.split("\t")
                if start == "@@":
                    if len(compare) > 0:
                        self.compare_entities.append(compare)
                    self.head_entities.append(desc.split())
                    compare = []
                else:
                    compare.append(desc.split())
            if len(compare) > 0:
                self.compare_entities.append(compare)


    def do_prepare(self, params, prepare):
        if 'similarity' in params:
            self.similarity = params.similarity
        else:  # Default similarity is cosine
            self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))

    def run(self, params, batcher):
        self.head_entities = [[None, None, None] + [item] for item in self.head_entities]
        head_entities_embedding = batcher(params, self.head_entities)[1][:, None, :]
        compare_entities_embedding = []
        for compare_entities in self.compare_entities:
            compare_entities_embedding.append(batcher(params, [[None, None, None] + [item] for item in compare_entities])[1][None, :, :])
        compare_entities_embedding = np.concatenate(compare_entities_embedding, 0)

        dot_prod = np.sum(head_entities_embedding * compare_entities_embedding, -1) # num_head * num_compare
        head_entities_norm = np.sqrt(np.sum(head_entities_embedding * head_entities_embedding, 2))
        compare_entities_norm = np.sqrt(np.sum(compare_entities_embedding * compare_entities_embedding, 2))
        cosine_similarities = dot_prod / head_entities_norm / compare_entities_norm
        cosine_similarities = cosine_similarities.reshape(-1)

        head_entities_count, compare_entities_count = compare_entities_norm.shape
        gold_scores = np.concatenate([np.arange(compare_entities_count)[None, ::-1] for i in range(head_entities_count)], 0).reshape(-1)


        results = {'pearson': pearsonr(cosine_similarities, gold_scores),
                    'spearman': spearmanr(cosine_similarities, gold_scores)}
        logging.debug('KORE relatedness dataset, Pearson = %.4f, \
            Spearman = %.4f\n' % (results["pearson"][0], results["spearman"][0]))

        return results
