#! /usr/bin/env python3

import sys
from image_proc import *
print(sys.argv)
test = ImageProc(sys.argv[1])
test.get_lines()


