#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import numpy as np
import pandas as pd

import GPUtil
from threading import Thread
import time

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
    index : bool
        | Write row names or not. (default : False)
    verbose : bool
        | Print fpath or not. (default : False)

    NOTE: other attributes follow the original documentation of pandas.DataFrame.
    '''

    if type(data) != 'pandas.core.frame.DataFrame':
        data = pd.DataFrame(data)

    makedir(fpath)
    writer = pd.ExcelWriter(fpath)
    data.to_excel(writer, 'Sheet1', index=index)
    writer.save()

    if verbose:
        print('Saved data as: {}'.format(fpath))


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


class GPUMonitor(Thread):
    '''
    A class to monitor the GPU status.
    Refer to "https://github.com/anderskm/gputil" and "https://data-newbie.tistory.com/561" for usages.

    Attributes
    ----------
    delay : float
        | An interval to display the GPU status.
    '''

    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True