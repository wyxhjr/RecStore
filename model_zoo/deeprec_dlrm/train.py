#!/usr/bin/env python3
import os
import sys

# Ensure project root is in sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Add third_party DeepRec path
DEEPrec_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../third_party/DeepRec/modelzoo/dlrm'))
if DEEPrec_ROOT not in sys.path:
    sys.path.insert(0, DEEPrec_ROOT)

if __name__ == '__main__':
    # Delegate to DeepRec's original entry
    import train as deeprec_train  # third_party/DeepRec/modelzoo/dlrm/train.py
    # DeepRec's train.py runs main via __main__ guard; emulate CLI execution
    deeprec_train.__name__ = '__main__'
    exec(open(os.path.join(DEEPrec_ROOT, 'train.py'), 'rb').read(), deeprec_train.__dict__)