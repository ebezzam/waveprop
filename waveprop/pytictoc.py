# -*- coding: utf-8 -*-

"""
Module with class TicToc to replicate the functionality of MATLAB's tic and toc.

Documentation: https://pypi.python.org/pypi/pytictoc
"""

__author__       = 'Eric Fields'
__version__      = '1.5.3'
__version_date__ = '2 August 2023'


from timeit import default_timer

class TicToc(object):
    
    """
    Replicate the functionality of MATLAB's tic and toc.
    
    #Methods
    TicToc.tic()       #start or re-start the timer
    TicToc.toc()       #print elapsed time since timer start
    TicToc.tocvalue()  #return floating point value of elapsed time since timer start
    
    #Attributes
    TicToc.start     #Time from timeit.default_timer() when t.tic() was last called
    TicToc.end       #Time from timeit.default_timer() when t.toc() or t.tocvalue() was last called
    TicToc.elapsed   #t.end - t.start; i.e., time elapsed from t.start when t.toc() or t.tocvalue() was last called
    """
    
    def __init__(self):
        """Create instance of TicToc class."""
        self.start   = float('nan')
        self.end     = float('nan')
        self.elapsed = float('nan')
        
    def tic(self):
        """Start the timer."""
        self.start = default_timer()
        
    def toc(self, msg='Elapsed time is', restart=False):
        """
        Report time elapsed since last call to tic().
        
        Optional arguments:
            msg     - String to replace default message of 'Elapsed time is'
            restart - Boolean specifying whether to restart the timer
        """
        self.end     = default_timer()
        self.elapsed = self.end - self.start
        print('%s %f seconds.' % (msg, self.elapsed))
        if restart:
            self.start = default_timer()
        
    def tocvalue(self, restart=False):
        """
        Return time elapsed since last call to tic().
        
        Optional argument:
            restart - Boolean specifying whether to restart the timer
        """
        self.end     = default_timer()
        self.elapsed = self.end - self.start
        if restart:
            self.start = default_timer()
        return self.elapsed
    
    def __enter__(self):
        """Start the timer when using TicToc in a context manager."""
        self.start = default_timer()
    
    def __exit__(self, *args):
        """On exit, pring time elapsed since entering context manager."""
        self.end = default_timer()
        self.elapsed = self.end - self.start
        print('Elapsed time is %f seconds.' % self.elapsed)
