__author__ = 'darkoa'

import unittest
import os
from os.path import normpath, dirname

import orange

class Tests1(unittest.TestCase):


    def test_eval_batch_avg_stg(self):
        import cf_noise.library as l
        # ARFF file
        f = normpath(dirname(__file__)) + os.sep + "glass.arff"

        data = None
        try:
            data = orange.ExampleTable(f)
        except Exception, e:
            print e

        perfs = [{'positives': [0,4,6],
                  'by_alg': [
                      {'name': 'Naive Bayes (Orange)', 'inds': [0,3,7]},
                      {'name': 'Random Forest-RF500 (Orange)', 'inds': [0,3,7]},
                      {'name': 'SVM (Orange)', 'inds': [0,3,5]}] }]


        inp_dict = {'perfs':perfs, 'beta':0.5}

        out_dict = l.eval_batch(inp_dict)
        self.assertGreaterEqual(len(out_dict.keys()), 1)

        # now connect this to avg_std:
        inp_dict = {'perf_results': out_dict['perf_results']}

        out_dict = l.avrg_std(inp_dict)
        self.assertGreaterEqual(len(out_dict.keys()), 1)



    def test_add_class_noise(self):
        import cf_noise.library as l
        # ARFF file
        f = normpath(dirname(__file__)) + os.sep + "glass.arff"

        data = None
        try:
            data = orange.ExampleTable(f)
        except Exception, e:
            print e
        inp_dict = {'data':data, 'noise_level':10.0, 'rnd_seed':1}

        out_dict = l.add_class_noise(inp_dict)


    def test_harf(self):
        """ Tests harf
        """

        import cf_noise.library as l
        inp_dict = {'agr_level':'70'}

        out_dict = l.harf(inp_dict)

        print len(out_dict.keys())


    def test_noise_rank(self):
        """
        """
        import cf_noise.library as l
        import cf_weka_local.classification as c

        learner = c.J48_learner()

        # ARFF file
        f = normpath(dirname(__file__)) + os.sep + "glass.arff"

        data = None
        try:
            data = orange.ExampleTable(f)
        except Exception, e:
            print e
        inp_dict = {'learner':learner, 'data':data, 'timeout':60.0, 'k_folds':10}

        out_dict = l.classification_filter(inp_dict, None)

        inp_dict = {'noise': [ out_dict['noise_dict'] ],
                    'data':data}

        l.noiserank(inp_dict)



    def test_two_filters(self):
        """ Tests saturation_filter (normal and prune) and classification_filter
        """

        import cf_noise.library as l
        import cf_weka_local.classification as c

        learner = c.J48_learner()

        # ARFF file
        f = normpath(dirname(__file__)) + os.sep + "glass.arff"

        data = None
        try:
            data = orange.ExampleTable(f)
        except Exception, e:
            print e


        inp_dict = {'data':data, 'satur_type':'normal'}
        out_dict = l.saturation_filter(inp_dict, None)

        self.assertGreaterEqual(len(out_dict.keys()), 1)

        inp_dict = {'data':data, 'satur_type':'prune'}
        out_dict = l.saturation_filter(inp_dict, None)


        self.assertGreaterEqual(len(out_dict.keys()), 1)

        inp_dict = {'learner':learner, 'data':data, 'timeout':60.0, 'k_folds':10}

        out_dict = l.classification_filter(inp_dict, None)

        self.assertGreaterEqual(len(out_dict.keys()), 1)

