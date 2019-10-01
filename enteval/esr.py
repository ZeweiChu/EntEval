from __future__ import absolute_import, division, unicode_literals

import os
import io
import numpy as np
import logging
import codecs

from scipy.stats import spearmanr, pearsonr
from enteval.utils import cosine

class WikiSRSEval(object):
    def __init__(self, taskpath, use_name=False, seed=1111):
        logging.debug('***** Transfer task : Entity Similarity and Relatedness *****\n\n')
        self.seed = seed
        self.use_name = use_name

        self.relate_labels, self.relate_entity1, self.relate_entity2 = self.loadFile(os.path.join(taskpath, "WikiSRS_relatedness.csv.pro"))
        self.sim_labels, self.sim_entity1, self.sim_entity2 = self.loadFile(os.path.join(taskpath, "WikiSRS_similarity.csv.pro"))

        self.relate_labels = np.array(self.relate_labels)
        self.sim_labels = np.array(self.sim_labels)

        self.samples = self.sim_entity1 + self.sim_entity2 + self.relate_entity1 + self.relate_entity2

    def loadFile(self, fpath):
        labels, entities1, entities2 = [], [], []
        with codecs.open(fpath, 'rb', 'utf-8') as f:
            for line in f:
                label, entity1, entity2, entity_desc1, entity_desc2 = line.strip().split("\t")
                labels.append(float(label))
                if self.use_name:
                    entities1.append(entity1.split())
                    entities2.append(entity2.split())
                else:
                    entities1.append(entity_desc1.split())
                    entities2.append(entity_desc2.split())
        return labels, entities1, entities2


    def do_prepare(self, params, prepare):
        if 'similarity' in params:
            self.similarity = params.similarity
        else:  # Default similarity is cosine
            self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))
        return prepare(params, self.samples)

    def calculate_similarity(self, params, batcher, ent1, ent2):
        assert len(ent1) == len(ent2), "entity 1 and entity 2 must have the same length"
        ent1 = [[None, None, None] + [item] for item in ent1]
        ent2 = [[None, None, None] + [item] for item in ent2]
        length = len(ent1)

        ent1_enc = []
        ent2_enc = []
        for i in range(0, length, params.batch_size):
            _, enc1 = batcher(params, ent1[i:i+params.batch_size])
            _, enc2 = batcher(params, ent2[i:i+params.batch_size])
            ent1_enc.append(enc1)
            ent2_enc.append(enc2)

        ent1_enc = np.vstack(ent1_enc)
        ent2_enc = np.vstack(ent2_enc)

        dot_prod = np.sum(ent1_enc * ent2_enc, -1)
        ent1_norm = np.sqrt(np.sum(ent1_enc ** 2, 1))
        ent2_norm = np.sqrt(np.sum(ent2_enc ** 2, 1))
        cosine_similarities = dot_prod / (ent1_norm + 1e-9) / (ent2_norm + 1e-9)
        cosine_similarities = cosine_similarities.reshape(-1)

        return cosine_similarities


    def run(self, params, batcher):
        relate_preds = self.calculate_similarity(params, batcher, self.relate_entity1, self.relate_entity2)
        sim_preds = self.calculate_similarity(params, batcher, self.sim_entity1, self.sim_entity2)


        results = {'relate_pearson': pearsonr(relate_preds, self.relate_labels),
                    'relate_spearman': spearmanr(relate_preds, self.relate_labels),
                    'sim_pearson': pearsonr(sim_preds, self.sim_labels),
                    'sim_spearman': spearmanr(sim_preds, self.sim_labels)}
        logging.debug('Wiki SRS dataset, Relatedness Pearson = %.4f, \
            Spearman = %.4f\n, Similarity Pearson = %.4f, \
            Spearman = %.4f\n' % (results["relate_pearson"][0], results["relate_spearman"][0], results["sim_pearson"][0], results["sim_spearman"][0]))

        return results

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
        self.samples = self.head_entities.copy()
        for entity in self.compare_entities:
            self.samples += entity


    def do_prepare(self, params, prepare):
        if 'similarity' in params:
            self.similarity = params.similarity
        else:  # Default similarity is cosine
            self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))
        return prepare(params, self.samples)

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
        cosine_similarities = dot_prod / (head_entities_norm + 1e-9) / (compare_entities_norm + 1e-9)
        cosine_similarities = cosine_similarities.reshape(-1)

        head_entities_count, compare_entities_count = compare_entities_norm.shape
        gold_scores = np.concatenate([np.arange(compare_entities_count)[None, ::-1] for i in range(head_entities_count)], 0).reshape(-1)


        results = {'pearson': pearsonr(cosine_similarities, gold_scores),
                    'spearman': spearmanr(cosine_similarities, gold_scores)}
        logging.debug('KORE relatedness dataset, Pearson = %.4f, \
            Spearman = %.4f\n' % (results["pearson"][0], results["spearman"][0]))

        return results
