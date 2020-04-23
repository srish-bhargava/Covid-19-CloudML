#!/usr/bin/env python

#Daniel Sabba ds1952 HW4 submission

from operator import itemgetter
import csv
import sys
import os

filename = os.environ['map_input_file'].split('/')[-1]
content=csv.reader(sys.stdin)

if filename == "tweets.csv":
     for line in content:
        line_content = list(line)
        print '%s\t%s\t%s\t%s' % (line_content[2], line_content[0],line_content[1],line_content[2])

if filename == "users.csv":
     for line in content:
        line_content = list(line)
        print '%s\t%s\t%s\t%s' % (line_content[0], line_content[0],line_content[1],line_content[2])
