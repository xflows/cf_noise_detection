import cf_noise_detection.utilities as u
import cf_data_mining.dataset as d

# ===================================================================
# HARF (HIGH AGREEMENT RANDOM FOREST)

def harf(input_dict):
    #import orngRF_HARF
    from cf_core.helpers import UnpicklableObject
    agrLevel = input_dict['agr_level']
    #data = input_dict['data']
    harfout = UnpicklableObject("cf_noise_detection.orngRF_HARF.HARFLearner(agrLevel ="+agrLevel+", name='HARF-"+str(agrLevel)+"')")
    harfout.addimport("import cf_noise_detection.orngRF_HARF")
    #harfLearner = orngRF_HARF.HARFLearner(agrLevel = agrLevel, name = "_HARF-"+agrLevel+"_")
    output_dict = {}
    output_dict['harfout']= harfout
    return output_dict

# CLASSIFICATION NOISE FILTER

def classification_filter(input_dict, widget):
    import cf_noise_detection.noiseAlgorithms4lib as nalg
    output_dict = {}
    # output_dict['noise_dict']= noiseAlgorithms4lib.cf_decide(input_dict, widget)

    orange_dataset = u.convert_dataset_from_scikit_to_orange(input_dict['data'])

    output_dict['noise_dict']= nalg.cf_decide(input_dict['learner'], orange_dataset, int(input_dict['k_folds']), widget=None)
    return output_dict

# SATURATION NOISE FILTER

def saturation_filter(input_dict, widget):
    import cf_noise_detection.noiseAlgorithms4lib as nalg

    orange_dataset = u.convert_dataset_from_scikit_to_orange(input_dict['data'])

    if not(input_dict['satur_type'] in ['normal', 'prune']):
        raise Exception("Only 'normal' or 'prune' allowed for 'satur_type'.")

    output_dict = {}
    output_dict['noise_dict']= nalg.saturation_type(orange_dataset, input_dict['satur_type'], widget)
    return output_dict

# NOISE RANK

def noiserank(input_dict):
    """Widget NoiseRank

    :param input_dict:
    :return:
    """
    allnoise = {}

    data = u.convert_dataset_from_scikit_to_orange(input_dict['data'])

    for item in input_dict['noise']:
        det_by = item['name']
        for i in item['inds']:
            if not allnoise.has_key(i):
                allnoise[i] = {}
                allnoise[i]['id'] = i
                allnoise[i]['class'] = data[int(i)].getclass().value
                allnoise[i]['by'] = []
            allnoise[i]['by'].append(det_by)
            print allnoise[i]['by']

    from operator import itemgetter
    outallnoise = sorted(allnoise.values(), key=itemgetter('id'))
    outallnoise.sort(compareNoisyExamples)

    output_dict = {}
    output_dict['allnoise'] = outallnoise
    output_dict['selection'] = {}
    return output_dict

def compareNoisyExamples(item1, item2):
    len1 = len(item1["by"])
    len2 = len(item2["by"])
    if len1 > len2: # reversed, want to have decreasing order
        return -1
    elif len1 < len2: # reversed, want to have decreasing order
        return 1
    else:
        return 0

def noiserank_select(postdata,input_dict, output_dict):
    try:
        output_dict['indices']= outselection = [int(i) for i in postdata['selected']]

        # data = input_dict['data']
        data = u.convert_dataset_from_scikit_to_orange(input_dict['data'])

        selection = [0]*len(data)
        for i in outselection:
            selection[i] = 1
        outdata = data.select(selection, 1)

        data_scikit = u.convert_dataset_from_orange_to_scikit(outdata)

        output_dict['selection'] = data_scikit
    except KeyError:
        output_dict['selection'] = None

    return output_dict


# EVALUATION OF NOISE DETECTION PERFORMANCE

def add_class_noise(input_dict):
    """Widget Add Class Noise
    """

    data_scikit = input_dict['data']
    if not(d.is_target_nominal(data_scikit)):
        raise Exception("Widget Add Class Noise accepts only datasets with nominal class!")

    data = u.convert_dataset_from_scikit_to_orange(data_scikit)

    import cf_noise_detection.noiseAlgorithms4lib as nalg
    noise_indices, orange_data = nalg.add_class_noise(data, input_dict['noise_level'], input_dict['rnd_seed'])

    data = u.convert_dataset_from_orange_to_scikit(orange_data)

    output_dict = {'noise_inds':noise_indices, 'noisy_data': data}

    return output_dict

def aggr_results(input_dict):
    """Widget Aggregate Detection Results

    :param input_dict:
    :return:
    """
    output_dict = {}
    output_dict['aggr_dict'] = { 'positives' : input_dict['pos_inds'], 'by_alg': input_dict['detected_inds']}
    return output_dict


def eval_batch(input_dict):
    """Widget "Evaluate Repeated Detection"
    """

    alg_perfs = input_dict['perfs']
    beta = float(input_dict['beta'])
    performances = []
    for exper in alg_perfs:
        noise = exper['positives']
        nds = exper['by_alg']

        performance = []
        for nd in nds:
            nd_alg = nd['name']
            det_noise = nd['inds']
            inboth = set(noise).intersection(set(det_noise))
            recall = len(inboth)*1.0/len(noise) if len(noise) > 0 else 0
            precision = len(inboth)*1.0/len(det_noise) if len(det_noise) > 0 else 0

            print beta, recall, precision
            if precision == 0 and recall == 0:
                fscore = 0
            else:
                fscore = (1+beta**2)*precision*recall/((beta**2)*precision + recall)
            performance.append({'name':nd_alg, 'recall': recall, 'precision' : precision, 'fscore' : fscore, 'fbeta': beta})

        performances.append(performance)

    output_dict = {}
    output_dict['perf_results'] = performances
    return output_dict

def eval_noise_detection(input_dict):
    """Widget "Evaluate Detection Algorithms"

    :param input_dict:
    :return:
    """
    noise = input_dict['noisy_inds']
    nds = input_dict['detected_noise']

    performance = []
    for nd in nds:
        nd_alg = nd['name']
        det_noise = nd['inds']
        inboth = set(noise).intersection(set(det_noise))
        recall = len(inboth)*1.0/len(noise) if len(noise) > 0 else 0
        precision = len(inboth)*1.0/len(det_noise) if len(det_noise) > 0 else 0
        beta = float(input_dict['f_beta'])
        print beta, recall, precision
        if precision == 0 and recall == 0:
            fscore = 0
        else:
            fscore = (1+beta**2)*precision*recall/((beta**2)*precision + recall)
        performance.append({'name':nd_alg, 'recall': recall, 'precision' : precision, 'fscore' : fscore, 'fbeta': beta})

    from operator import itemgetter
    output_dict = {}
    output_dict['nd_eval'] = sorted(performance, key=itemgetter('name'))
    return output_dict

# ENSEMBLE

def noise_detect_ensemble(input_dict):
    """ Noise detection ensemble

    :param input_dict:
    :return:
    """

    import math
    ens = {}
    data_inds = input_dict['data_inds']
    ens_type = input_dict['ens_type']

    for item in data_inds:
        #det_by = item['detected_by']
        for i in item['inds']:
            if not ens.has_key(i):
                ens[i] = 1
            else:
                ens[i] += 1

    ens_out = {}
    ens_out['name'] = input_dict['ens_name']
    ens_out['inds'] = []
    n_algs = len(data_inds)
    print ens_type
    if ens_type == "consensus":
        ens_out['inds'] = sorted([x[0] for x in ens.items() if x[1] == n_algs])
    else: # majority
        ens_out['inds'] = sorted([x[0] for x in ens.items() if x[1] >= math.floor(n_algs/2+1)])

    output_dict = {}
    output_dict['ens_out'] = ens_out
    return output_dict


# VISUALIZATIONS

def eval_to_table(input_dict):
    """Widget Evaluation Results to Table"""
    return {}

# def pr_space(input_dict):
#     return {}
#
# def eval_bar_chart(input_dict):
#     return {}
#
#
# def data_table(input_dict):
#     return {}
#
# def data_info(input_dict):
#     return {}
#
# def definition_sentences(input_dict):
#     return {}
#
# def term_candidates(input_dict):
#     return {}

