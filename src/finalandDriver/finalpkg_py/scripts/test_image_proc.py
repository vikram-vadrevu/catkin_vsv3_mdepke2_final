#! /usr/bin/env python3

import sys
#from finalandDriver.finalpkg_py.scripts.image_proc import ImageProc
from image_proc import *
print(sys.argv)
test = ImageProc(sys.argv[1])
test.get_lines()


