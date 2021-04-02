#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import pandas as pd


def makedir(fpath):
    '''
    A method to make directory for the given file path.

    Attributes
    ----------
    fpath : str
        | A file path.
    '''

    if fpath.endswith('/'):
        os.makedirs(fpath, exist_ok=True)
    else:
        os.makedirs('/'.join(fpath.split('/')[:-1]), exist_ok=True)


def export_xlsx(data, fpath, index=False, verbose=False):
    '''
    A method to export excel of the given data.

    Attributes
    ----------
    data : dict
        | A dictionary of which column lengths are the same.
    fpath : str
        | The file path to export the xlsx.
    index : Boolean
        | Write row names or not. (default : False)
    verbose : Boolean
        | Print fpath or not. (default : False)

    NOTE: other attributes follow the original documentation of pandas.DataFrame.
    '''

    if type(data) != 'pandas.core.frame.DataFrame':
        data = pd.DataFrame(data)

    makedir(fpath)
    writer = pd.ExcelWriter(fpath)
    data.to_excel(writer, "Sheet1", index=index)
    writer.save()

    if verbose:
        print("Saved data as: {}".format(fpath))


def f1_score(p, r):
    '''
    A method to calculate f1 score.

    Attributes
    ----------
    p : float
        | precision (None-zero)
    r : float
        | recall (None-zero)
    '''

    if p != 0 or r != 0:
        return (2*p*r)/(p+r)
    else:
        return 0



def accuracy(tp, tn, fp, fn):
    '''
    A method to calculate accuracy

    Attributes
    ----------
    tp : int
        | true positive
    tn : int
        | true negative
    fp : int
        | false positive
    fn : int
        | false negative
    '''

    return (tp+tn)/(tp+tn+fp+fn)