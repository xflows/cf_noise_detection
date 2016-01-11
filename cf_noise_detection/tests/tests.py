__author__ = 'daleksovski'

import unittest

import cf_data_mining.utilities as ut


class Tests1(unittest.TestCase):
    def test_add_class_noise(self):
        import cf_noise_detection.library as l

        data = ut.load_UCI_dataset("iris")
        inp_dict = {'data': data, 'noise_level': 10.0, 'rnd_seed': 1}

        out_dict = l.add_class_noise(inp_dict)
        self.assertGreaterEqual(len(out_dict["noise_inds"]), 1)

    def test_harf(self):
        """ Tests harf
        """

        import cf_noise_detection.library as l
        inp_dict = {'agr_level': '70'}

        out_dict = l.harf(inp_dict)

        self.assertTrue(out_dict.has_key("harfout"))

    def test_noise_rank(self):
        """Tests noise rank widget
        """
        import cf_noise_detection.library as l
        import cf_weka.classification as c

        learner = c.j48()

        data = ut.load_UCI_dataset("iris")

        inp_dict = {'learner': learner, 'data': data, 'timeout': 60.0, 'k_folds': 10}

        out_dict = l.classification_filter(inp_dict, None)

        inp_dict = {'noise': [out_dict['noise_dict']],
                    'data': data}

        out_dict = l.noiserank(inp_dict)

        self.assertGreaterEqual(len(out_dict['allnoise']), 1)

    def test_two_filters(self):
        """ Tests saturation_filter (normal and prune) and classification_filter
        """

        import cf_noise_detection.library as l
        import cf_weka.classification as c

        learner = c.j48()

        data = ut.load_UCI_dataset("iris")

        inp_dict = {'data': data, 'satur_type': 'normal'}
        out_dict = l.saturation_filter(inp_dict, None)

        self.assertGreaterEqual(len(out_dict.keys()), 1)

        inp_dict = {'data': data, 'satur_type': 'prune'}
        out_dict = l.saturation_filter(inp_dict, None)

        self.assertGreaterEqual(len(out_dict.keys()), 1)

        inp_dict = {'learner': learner, 'data': data, 'timeout': 60.0, 'k_folds': 10}

        out_dict = l.classification_filter(inp_dict, None)

        self.assertGreaterEqual(len(out_dict.keys()), 1)
