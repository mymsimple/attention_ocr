#!/usr/bin/env python
import time
def timestamp_s():
    s = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    return s
