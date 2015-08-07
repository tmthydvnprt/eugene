"""
Util.py
"""

import sys
import numpy as np

def rmse(predicted, truth):
    """return the mean square error"""
    try:
        if np.isnan(predicted).any() or predicted.shape != truth.shape:
            result = np.inf
        else:
            result = np.sqrt(((predicted - truth) ** 2).mean())
    except TypeError:
        result = np.inf
    return result

class ProgressBar(object):
    """implements a comand-line progress bar"""

    def __init__(self, iterations):
        """create a progress bar"""
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 40
        self.__update_amount(0)

    def animate(self, iterate):
        """animate progress"""
        print '\r', self,
        sys.stdout.flush()
        self.update_iteration(iterate + 1)
        return self

    def update_iteration(self, elapsed_iter):
        """increment progress"""
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar = '%s  %s of %s complete' % (self.prog_bar, elapsed_iter, self.iterations)
        return self

    def __update_amount(self, new_amount):
        """update amout of progress"""
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[%s%s]' % ((self.fill_char * num_hashes), ' ' * (all_full - num_hashes))
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%s%%' % (percent_done)
        self.prog_bar = '%s%s%s' % (self.prog_bar[0:pct_place], pct_string, self.prog_bar[pct_place + len(pct_string):])
        return self

    def __str__(self):
        """string representation"""
        return str(self.prog_bar)
