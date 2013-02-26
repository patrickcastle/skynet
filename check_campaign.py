#!/usr/bin/env python

import sys
import time_series

campaign_id = int(sys.argv[1])
if time_series.check_campaign(campaign_id):
    print true
else:
    print false
