# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''

Generic sentence evaluation scripts wrapper

'''
from __future__ import absolute_import, division, unicode_literals

from enteval import utils
from enteval.cap import CAPEval
from enteval.ert import ERTEval
from enteval.efp import EFPEval
from enteval.ed import RareEval, ConllYagoEval 
from enteval.et import ETEval
from enteval.cerp import CERPEval
from enteval.esr import WikiSRSEval, KOREEval


class SE(object):
    def __init__(self, params, batcher, prepare=None):
        # parameters
        params = utils.dotdict(params)
        params.usepytorch = True if 'usepytorch' not in params else params.usepytorch
        params.seed = 1111 if 'seed' not in params else params.seed

        params.batch_size = 128 if 'batch_size' not in params else params.batch_size
        params.nhid = 0 if 'nhid' not in params else params.nhid
        params.kfold = 5 if 'kfold' not in params else params.kfold

        if 'classifier' not in params or not params['classifier']:
            params.classifier = {'nhid': 0}

        assert 'nhid' in params.classifier, 'Set number of hidden units in classifier config!!'

        self.params = params

        # batcher and prepare
        self.batcher = batcher
        self.prepare = prepare if prepare else lambda x, y: None

        self.list_tasks = ['CAPsame', 'CAPnext', 'ER', 'EFP', 'ET', 'CERP', 'Rare', 'ConllYago', 'WikiSRS', 'KORE' ]

    def eval(self, name):
        # evaluate on evaluation [name], either takes string or list of strings
        if (isinstance(name, list)):
            self.results = {x: self.eval(x) for x in name}
            return self.results

        tpath = self.params.task_path
        assert name in self.list_tasks, str(name) + ' not in ' + str(self.list_tasks)

        # Entity Evaluation Tasks
        if name == 'CAPsame':
            self.evaluation = CAPEval(tpath + '/CAP/same', seed=self.params.seed)
        elif name == 'CAPnext':
            self.evaluation = CAPEval(tpath + '/CAP/next', seed=self.params.seed)
        elif name == 'ERT':
            self.evaluation = ERTEval(tpath + '/ERT/', seed=self.params.seed)
        elif name == 'EFP':
            self.evaluation = EFPEval(tpath + '/EFP/', use_ctx=False, seed=self.params.seed)
        elif name == 'ET':
            self.evaluation = ETEval(tpath + '/ET/', seed=self.params.seed)
        elif name == 'CERP':
            self.evaluation = CERPEval(tpath + '/CERP/', seed=self.params.seed)
        elif name == 'Rare':
            self.evaluation = RareEval(tpath + '/rare/', seed=self.params.seed)
        elif name == 'ConllYago':
            self.evaluation = ConllYagoEval(tpath + '/conll-yago/', seed=self.params.seed)
        elif name == 'KORE':
            self.evaluation = KOREEval(tpath + '/KORE/', seed=self.params.seed)
        elif name == 'WikiSRS':
            self.evaluation = WikiSRSEval(tpath + '/wikisrs/', seed=self.params.seed)

        self.params.current_task = name
        self.evaluation.do_prepare(self.params, self.prepare)

        self.results = self.evaluation.run(self.params, self.batcher)

        return self.results
