from __future__ import print_function

import sys
if(sys.version_info < (3, 0)):
    print("error: Python3 required")
    sys.exit(1)
    
import os

def replace(dirs, old, new):
    for dir in dirs:
        for root, directories, filenames in os.walk(dir):
            for filename in filenames:
                s = ""
                with open(os.path.join(root, filename), mode='r+') as f:
                    s = f.read()    
                    for o, n in zip(old, new):
                      s = s.replace(o, n)

                new_filename = filename
                for o, n in zip(old, new):
                    new_filename = new_filename.replace(o, n)

                with open(os.path.join(root, new_filename), mode='w+') as f:
                    f.write(s)
                  
if __name__ == '__main__':
    dir = ["./gtclang"]

    old = ["Gsl", "gsl", "GSL"]
    new = ["Dawn", "dawn", "DAWN"]
    
    # old = ["GCD", "gcd"]
    # new = ["GTCLANG", "gtclang"]
    replace(dir, old, new)
