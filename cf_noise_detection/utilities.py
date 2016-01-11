import cStringIO
import string

import cf_weka.common as c
import cf_weka.utilities as u


def convert_dataset_from_scikit_to_orange(dataset):
    """Converts dataset from a scikit Bunch format to an Orange Data Table

    :param dataset: dataset in bunch format
    :return:
    """

    arff_contents = u.export_dataset_to_arff(dataset)

    import orange

    # import tempfile;    f = tempfile.NamedTemporaryFile(delete=False,suffix='.arff')
    # f.write(arff_contents);         f.close()

    tmp = c.TemporaryFile(suffix='.arff')
    tmp.writeString(arff_contents)

    orange_dataset = orange.ExampleTable(tmp.name)

    return orange_dataset

def convert_dataset_from_orange_to_scikit(dataset):
    """Converts dataset from an Orange Data Table to scikit Bunch format

    :param dataset:
    :return:
    """

    arff_str = to_arff_string(dataset).getvalue()
    dataset_new = u.import_dataset_from_arff(arff_str)

    return dataset_new


def to_arff_string(table, try_numericize=0):#filename,table,try_numericize=0):
    """ Converts an Orange data table to ARFF string

    Args:
        table:
        try_numericize:

    Returns:

    """


    t = table
    #if filename[-5:] == ".arff":
     #   filename = filename[:-5]
    #print filename
    f = cStringIO.StringIO()
    f.write('@relation %s\n'%t.domain.classVar.name)
    # attributes
    ats = [i for i in t.domain.attributes]
    ats.append(t.domain.classVar)
    for i in ats:
        real = 1
        if i.varType == 1:
            if try_numericize:
                # try if all values numeric
                for j in i.values:
                    try:
                        x = string.atof(j)
                    except:
                        real = 0 # failed
                        break
            else:
                real = 0
        iname = str(i.name)
        if string.find(iname," ") != -1:
            iname = "'%s'"%iname
        if real==1:
            f.write('@attribute %s real\n'%iname)
        else:
            f.write('@attribute %s { '%iname)
            x = []
            for j in i.values:
                s = str(j)
                if string.find(s," ") == -1:
                    x.append("%s"%s)
                else:
                    x.append("'%s'"%s)
            for j in x[:-1]:
                f.write('%s,'%j)
            f.write('%s }\n'%x[-1])

    # examples
    f.write('@data\n')
    for j in t:
        x = []
        for i in range(len(ats)):
            s = str(j[i])
            if string.find(s," ") == -1:
                x.append("%s"%s)
            else:
                x.append("'%s'"%s)
        for i in x[:-1]:
            f.write('%s,'%i)
        f.write('%s\n'%x[-1])

    return f