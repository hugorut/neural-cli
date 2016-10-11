from __future__ import absolute_import, division, print_function
import sys

class Writer():
    """a slim wrapper around the sys package that provides
    an interface to implement for differing write output"""
    
    def write(self, characters, overwrite=True):
        if overwrite:
            whitespace = ' ' * 20
            sys.stdout.write('\r' + str(characters) + whitespace)
        else:
            sys.stdout.write(str(characters) + '\n')
        
        sys.stdout.flush()
