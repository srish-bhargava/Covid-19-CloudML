#!/usr/bin/python

import sys, os
arguments = sys.argv

command='/usr/bin/hadoop jar /usr/lib/hadoop/contrib/streaming/hadoop-streaming-1.0.3.16.jar -file join_map.py -mapper join_map.py -file join_reduce.py -reducer join_reduce.py -input '+ arguments[1] + ' -input '+ arguments[2]+ ' -output '+ arguments[3]

print command

os.system(command)