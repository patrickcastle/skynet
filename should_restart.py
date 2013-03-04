#/usr/bin/env python

import sys
import time_series

res = time_series.check_campaign(int(sys.argv[1]))
if True == res:
    print "true"
else:
    print "false"
