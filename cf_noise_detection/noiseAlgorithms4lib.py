import random

import cf_data_mining.classifier as c
import orange
import orngTree
from cf_core.helpers import UnpicklableObject

from cf_noise_detection.utilities import convert_dataset_from_orange_to_scikit


def add_class_noise(data, noise_level, rnd_seed):
    """adds class Noise

    :param data: Orange dataset
    :param noise_level:
    :param rnd_seed:
    :return:
    """

    meta_noisy = orange.EnumVariable("noise", values=["no", "yes"])
    mid = orange.newmetaid()
    while mid in data.domain.getmetas().keys():
        mid = orange.newmetaid()
    data.domain.addmeta(mid, meta_noisy)
    data.addMetaAttribute("noise", "no")
    # Generate random indices for noise insertion
    percent = float(noise_level)/100
    try:
        rnds = int(rnd_seed)
    except:
        rnds = 0
    print "Random Seed:", rnds
    orange.setrandseed(rnds)
    noise_indices = random.sample(range(len(data)), int(round(percent*len(data))))
    #print "Amount of added noise:", percent*100, "percent (", len(noise_indices), "examples ):"
    #print "Random indices for added noise:", noise_indices
    className = data.domain.classVar.name
    #print "Class name:", className
    for index in noise_indices:
        data[index]["noise"] = "yes"
        temp = data[index][className]
##        if len(data.domain.classVar.values) > 2:
        # random value + check if it is diferent from the current one
        new_label = data.domain.classVar.randomvalue()
        while new_label == temp:
            new_label = data.domain.classVar.randomvalue()
        data[index][className] = new_label
##        else:
##            # switch the class value
##            data[index][className] = data.domain.classVar.nextvalue(data[index][className])
        #print "\t", temp, "changed to:", data[index].getclass(), "(", index, ")"
    #print "\n"
    noise_indices.sort()
    return noise_indices, data

def add_meta_id(data):
    meta_id = orange.FloatVariable("meta_id")
    mid = orange.newmetaid()
    while mid in data.domain.getmetas().keys():
        mid = orange.newmetaid()
    data.domain.addmeta(mid, meta_id)
    for i in range(len(data)):
        data[i][meta_id] = i


def cf_decide(learner, orange_dataset, k_folds, widget):
    """Classification filter decide

    :param learner: Classifier object
    :param orange_dataset:
    :param k_folds:
    :param widget:
    :return:
    """

    # somelearner = input_dict['learner']
    print learner
    # SWITCH TO PROCESSING WITH WEKA CLASSIFIERS    
    if isinstance(learner, c.Classifier):

        name = learner.print_classifier()

        return cf_run(learner,
                      orange_dataset,
                      k_folds,
                      name,
                      widget)
    else:
        return cf_run_harf(learner, orange_dataset, k_folds, widget)
    # else:
    #     raise Exception("Provided learner is in an unsupported format", str(learner))


def cf_run(learner, data, k_folds, name, widget=None):
    """Runs a classification filter

    :param learner: WekaClassifier
    :param data: Orange dataset
    :param k_folds:
    :param name:
    :param timeout:
    :param widget:
    :return:
    """

    somelearner = learner
    print somelearner

    noisyIndices = []
    selection = orange.MakeRandomIndicesCV(data, folds=k_folds)
    count_noisy = [0]*k_folds
    for test_fold in range(k_folds):
        # train_data = wsutil.client.arff_to_weka_instances(arff = train_arffstr, class_index = data.domain.index(data.domain.classVar))['instances']
        train_data = convert_dataset_from_orange_to_scikit( data.select(selection, test_fold, negate=1) )

        test_inds = [i for i in range(len(selection)) if selection[i] == test_fold ]
        # test_data = wsutil.client.arff_to_weka_instances(arff = test_arffstr, class_index = data.domain.index(data.domain.classVar))['instances']
        test_data = convert_dataset_from_orange_to_scikit( data.select(selection, test_fold) )

        #print "\t\t", "Learned on", len(train_data), "examples"
        #file.flush()

        print "before cl build"
        # classifier = wseval.client.build_classifier(learner = somelearner, instances = train_data)['classifier']
        learner.build_classifier(train_data)
        print "after cl build"

        # eval_test_data = wseval.client.apply_classifier(classifier = classifier, instances = test_data)
        scikit_dataset_predicted = learner.apply_classifier(test_data)

        print "after apply"

        for i in range(len(scikit_dataset_predicted.target)):
            #print "Test data length:", len(test_data), "Test inds length:", len(test_inds), "Eval Test data length:", len(eval_test_data)  
            # print i, "v for zanki", eval_test_data[i]['classes'], data[test_inds[i]].getclass()
            # if eval_test_data[i]['classes'] != unicode(data[test_inds[i]].getclass()):

            if scikit_dataset_predicted.target[i] != scikit_dataset_predicted.targetPredicted[i]:
                # selection_filter[int(example[meta_id].value)] = 0
                noisyIndices.append(test_inds[i])
                count_noisy[test_fold] += 1
        # END test_data
        if not(widget is None):
            widget.progress = int((test_fold+1)*1.0/k_folds*100)
            widget.save()
    # END test_fold
    return {'inds': sorted(noisyIndices), 'name': get_weka_name(name)}

def cf_run_harf(learner, data_orange, k_folds, widget=None):
    """Classification filter for HARF learner

    :param learner:
    :param data_orange:
    :param k_folds:
    :param widget:
    :return:
    """

    somelearner = learner
    print "Before generate"
    learner = somelearner if not isinstance(somelearner,UnpicklableObject) else somelearner.generate()
    print "After generate"
    # data_orange = input_dict['data_orange']
    print len(data_orange)
    add_meta_id(data_orange)
    print 'Before for loop'
    k = k_folds
    noisyIndices = []
    selection = orange.MakeRandomIndicesCV(data_orange, folds=k)
    count_noisy = [0]*k
    print 'Before for loop'
    for test_fold in range(k):
        train_data = data_orange.select(selection, test_fold, negate=1)
        test_data = data_orange.select(selection, test_fold)
        #print "\t\t", "Learned on", len(train_data), "examples"
        #file.flush()
        print 'Before classifier construction'
        #print learner.hovername if learner.hovername != None else "ni hovernamea"
        classifier = learner(train_data)
        print 'After classifier construction'
        for example in test_data:
            exclassified = classifier(example)
            if exclassified != None and exclassified != example.getclass():
                # selection_filter[int(example[meta_id].value)] = 0
                noisyIndices.append(int(example["meta_id"].value))
                count_noisy[test_fold] += 1
        # END test_data
        if not(widget is None):
            widget.progress = int((test_fold+1)*1.0/k*100)
            widget.save()
    # END test_fold
    return {'inds': sorted(noisyIndices), 'name': learner.name}


def saturation_type(dataset, satur_type='normal', widget=None):
    """Saturation filter

    :param dataset: Orange dataset
    :param satur_type: 'normal' or 'prune'
    :param widget:
    :return:
    """

    add_meta_id(dataset)
    if not(widget==None):
        widget.progress = 0
        widget.save()
    data_len = len(dataset)
    #k = data_len/2
    progress_steps = (3*data_len**2 + 2*data_len)/8 # provided max allowed iter steps (k) = data_len/2
    if satur_type == 'prune':
        if not dataset.hasMissingValues():
            return prune_sf(dataset, 1, progress_steps, widget)
        else:
            raise Exception("Pre-pruned saturation filtering requires data WITHOUT missing values!")
    else:
        return saturation(dataset, widget)
    
def cmplx(set):
    classifier = orngTree.TreeLearner(set, sameMajorityPruning=1, mForPruning=0)
    return orngTree.countNodes(classifier)

def find_noise(data):
    n = len(data)
    noisiest = []
    gE = cmplx(data)
    print "\t\t", "Classifier complexity:", gE, "nodes"
    #file.flush()
    min = gE
    for i in range(n):
        selection = [1]*n
        selection[i] = 0
        Ex = data.select(selection)
        if len(Ex)== 0:
            print "\t\t", "Saturation Filtering FAILED!"
            #file.flush()
            return [0, []]
        else:
            gEx = cmplx(Ex)
        if gEx < min:
            noisiest = [i]
            min = gEx
            print "\t\t", "(%s." % int(data[i]["meta_id"]),"example excluded) Subset complexity:", gEx, "nodes"#, "(%s)" % data[i]["noise"].value
            #file.flush()
            #print data[i]
        elif gEx != gE and gEx == min:
            noisiest.append(i)
            print "\t\t", "(%s." % int(data[i]["meta_id"]),"example excluded) Subset complexity:", gEx, "nodes"#, "(%s)" % data[i]["noise"].value
            #file.flush()
            #print data[i]
    if noisiest != []:
        return [0, noisiest]
    else:
        return [1, []]

def saturation(dataset, widget):
    """Saturation

    :param dataset: Orange dataset
    :param widget:
    :return:
    """


    #dataset = input_dict['data']
    print "\t","Saturation Filtering:"
    #file.flush()
    noisyA = orange.ExampleTable(dataset.domain)
    data_len = len(dataset)
    k = data_len/2
    progress_steps = (3*data_len**2 + 2*data_len)/8 # provided max allowed iter steps (k) = data_len/2
    if not(widget==None):
        prog_sum = widget.progress
    workSet = orange.ExampleTable(dataset)
    while k != 0:
        n = len(workSet)
        satfilter = find_noise(workSet)
        if satfilter == [1,[]]:
            print "\t\t", satfilter
            if not(widget==None):
                widget.progress = 100
                widget.save()
            break
        else:
            noisyExmpls = satfilter[1]
            #print noisyExmpls
            selection = [0]*n
            choose = random.choice(noisyExmpls)
            print "\t\t", "Randomly choose one noisy example among:", len(noisyExmpls),\
            #      "(%s. is added noise: %s)" % (int(workSet[choose]["meta_id"]), workSet[choose]["noise"].value)
            #file.flush()
            selection[choose] = 1
            noisyA.extend(workSet.select(selection))
            workSet = workSet.select(selection, negate=1)

            if not(widget==None):
                prog_sum += n*1.0/progress_steps*100
                widget.progress = int(prog_sum)
                widget.save()
                print "widget prog: ", widget.progress, "n: ", n, "progress_steps:", progress_steps, "prog_sum:", prog_sum
        k -= 1
    print "\t\t", "Found:", len(noisyA), "examples.\n"
    #file.flush()
    noisyIndices = []
    for ex in noisyA:
        noisyIndices.append(int(ex["meta_id"].value))
    #return [noisyA, workSet]
    #return [noisyIndices, workSet]
    return {"inds" : sorted(noisyIndices), "name" : "SF"}

def findPrunableNoisy(node, minExmplsInLeaf):
    toPrune = []
    print "in find, toPrune:", toPrune
    if isinstance(node, orange.TreeNode):
        #print "Bu!"
        if node and node.branchSelector:
            #print "Bu111!"
            for branch in node.branches:
                if branch == None:
                    continue
                else:
                    if len(branch.examples) > minExmplsInLeaf + 0.5:
                        bla = findPrunableNoisy(branch, minExmplsInLeaf)
                        toPrune.extend(bla)
                    else:
                        print "Zapisal za brisanje"
                        for ex in branch.examples:
                            toPrune.append(int(ex["meta_id"].value))
            return toPrune
        return []
    else:
        raise TypeError, "TreeNode expected"

def exclude_pruned(dataset, classifier, minExmplsInLeaf):
    print "in exclude"
    toPrune = findPrunableNoisy(classifier.tree, minExmplsInLeaf)
    unique_items(toPrune)
    print "\t\t", "Leaves with", minExmplsInLeaf, "or less examples will be pruned."
    print "\t\t", "IDs of examples excluded by pruning:", toPrune
    #file.flush()
    #noisyA = orange.ExampleTable(dataset.domain)
    n = len(dataset)
    selection = [0]*n
    for index in toPrune:
        selection[index] = 1
    #noisyA.extend(dataset.select(selection))
    workSet = dataset.select(selection, negate=1)
    #return [noisyA, dataset]
    return [toPrune, workSet]
    
def unique_items(list):
    list.sort()
    k = 0
    while k < len(list)-1:
        if list[k+1] == list[k]:
            del list[k+1]
        else:
            k += 1

def prune_sf(data, minExmplsInLeaf, progress_steps, widget=None):
    """Prune Saturation Filter

    :param data:
    :param minExmplsInLeaf:
    :param progress_steps:
    :param widget:
    :return:
    """

    print "\t", "Pruning + Saturation Filter:"
    #file.flush()
    classifier = orngTree.TreeLearner(data, sameMajorityPruning=1, mForPruning=0, storeExamples=1)
    print "\t\t", "Classifier complexity:\t", orngTree.countNodes(classifier), "nodes."
    #file.flush()
##    [noisyA, dataset] = exclude_pruned(data, classifier, minExmplsInLeaf)
    [noisePruned, dataset] = exclude_pruned(data, classifier, minExmplsInLeaf)
    print "\t\t", len(noisePruned), "example(s) were excluded by pruning."
    #file.flush()
    classifier2 = orngTree.TreeLearner(dataset, sameMajorityPruning=1, mForPruning=0, storeExamples=1)
    print "\t\t", "Pruned Classifier complexity:", orngTree.countNodes(classifier2), "nodes. "
    #file.flush()
    # Saturation filtering
##    [noisy_data, filtered_data] = saturation(dataset, "tree")
    
    n = len(data)
    #widget.progress = int(len(noisePruned)*1.0/len(data)*100)
    if not(widget==None):
        widget.progress = int(sum([n-i for i in range(len(noisePruned))])*1.0/progress_steps*100)
        widget.save()
        print "progress:", widget.progress

    #[noiseSF, filtered_data] = saturation(dataset, widget)#, "tree")
    noiseSF = saturation(dataset, widget)#, "tree")
    #print "\t\t", "Size of filtered dataset:", len(filtered_data)
    print "\t\t", "Noisy examples (", len(noiseSF["inds"])+len(noisePruned),"(",len(noisePruned),"pruned,",\
          len(noiseSF["inds"]), "SF ))\n"#: (class, id)"
    #file.flush()
    #noisy_data.sort(meta_id)
    #noiseSF.sort()
    # Merge both obtained sets of noisy examples
    #noisyA.extend(noisy_data)
    noisePruned.extend(noiseSF["inds"])
    #return noisyA
    return {"inds" : sorted(noisePruned), "name" : "PruneSF"}
    #return noisePruned
    
    

def get_weka_name(name):
    #print name
    if name == None:
        return 'Multilayer Perceptron (Weka)'
    elif name.startswith('No '):
        return 'J48 (Weka)'
    elif name.startswith('Naive Bayes') or name.startswith('Random forest'):
        return "".join([name.split()[0], ' ', name.split()[1], ' (Weka)'])
    else:
        return name.split()[0].rstrip(':') + ' (Weka)'