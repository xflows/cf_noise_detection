# ===================================================================
# HARF (HIGH AGREEMENT RANDOM FOREST)

def harf(input_dict):
    #import orngRF_HARF
    from cf_base.helpers import UnpicklableObject
    agrLevel = input_dict['agr_level']
    #data = input_dict['data']
    harfout = UnpicklableObject("orngRF_HARF.HARFLearner(agrLevel ="+agrLevel+", name='HARF-"+str(agrLevel)+"')")
    harfout.addimport("import orngRF_HARF")
    #harfLearner = orngRF_HARF.HARFLearner(agrLevel = agrLevel, name = "_HARF-"+agrLevel+"_")
    output_dict = {}
    output_dict['harfout']= harfout
    return output_dict

# CLASSIFICATION NOISE FILTER

def classification_filter(input_dict, widget):
    import cf_noise.noiseAlgorithms4lib as nalg
    output_dict = {}
    # output_dict['noise_dict']= noiseAlgorithms4lib.cfdecide(input_dict, widget)
    output_dict['noise_dict']= nalg.cfdecide(input_dict, widget=None)
    return output_dict

# SATURATION NOISE FILTER

def saturation_filter(input_dict, widget):
    import cf_noise.noiseAlgorithms4lib as nalg
    output_dict = {}
    output_dict['noise_dict']= nalg.saturation_type(input_dict['data'], widget)
    return output_dict

# NOISE RANK

def noiserank(input_dict):
    allnoise = {}
    data = input_dict['data']
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
        data = input_dict['data']
        selection = [0]*len(data)
        for i in outselection:
            selection[i] = 1
        outdata = data.select(selection, 1)
        output_dict['selection'] = outdata
    except KeyError:
        output_dict['selection'] = None

    return output_dict


# EVALUATION OF NOISE DETECTION PERFORMANCE

def add_class_noise(input_dict):
    """
    """
    import cf_noise.noiseAlgorithms4lib as nalg
    output_dict = nalg.addClassNoise(input_dict['data'], input_dict['noise_level'], input_dict['rnd_seed'])
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

def avrg_std(input_dict):
    """Widget "Average and Standard Deviation" which for some reason is missing from source.ijs.si
    -> to be connected on the left using widget "Evaluate Repeated Detection" (eval_batch)
    """
    perf_results = input_dict['perf_results']
    stats = {}
    # Aggregate performance results
    n = len(perf_results)
    for i in range(n):
        for item in perf_results[i]:
            alg = item['name']
            if not stats.has_key(alg):
                stats[alg] = {}
                stats[alg]['precisions'] = [item['precision']]
                stats[alg]['recalls'] = [item['recall']]
                stats[alg]['fscores'] = [item['fscore']]
                stats[alg]['fbeta'] = item['fbeta']
            else:
                stats[alg]['precisions'].append(item['precision'])
                stats[alg]['recalls'].append(item['recall'])
                stats[alg]['fscores'].append(item['fscore'])

            # if last experiment: compute averages
            if i == n-1:
                stats[alg]['avrg_pr'] = reduce(lambda x,y: x+y, stats[alg]['precisions'])/n
                stats[alg]['avrg_re'] = reduce(lambda x,y: x+y, stats[alg]['recalls'])/n
                stats[alg]['avrg_fs'] = reduce(lambda x,y: x+y, stats[alg]['fscores'])/n

    # Compute Standard Deviations
    import numpy
    avrgstdout = []
    print stats
    for alg, stat in stats.items():
        avrgstdout.append({'name': alg, 'precision': stat['avrg_pr'], 'recall': stat['avrg_re'],
                           'fscore' : stat['avrg_fs'],
                           'fbeta'  : stat['fbeta'],
                           'std_pr' : numpy.std(stat['precisions']),
                           'std_re' : numpy.std(stat['recalls']),
                           'std_fs' : numpy.std(stat['fscores']) })

    from operator import itemgetter
    output_dict = {}
    output_dict['avrg_w_std'] = sorted(avrgstdout, key=itemgetter('name'))
    return output_dict

# VISUALIZATIONS

def pr_space(input_dict):
    return {}

def eval_bar_chart(input_dict):
    return {}

def eval_to_table(input_dict):
    return {}

def data_table(input_dict):
    return {}

def data_info(input_dict):
    return {}

def definition_sentences(input_dict):
    return {}

def term_candidates(input_dict):
    return {}

