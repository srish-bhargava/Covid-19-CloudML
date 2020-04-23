#!/usr/bin/env python

#Daniel Sabba ds1952 HW4 submission

from operator import itemgetter
import sys
dict={}

for line in sys.stdin:
    line_content = line.strip().split("\t")
    line_index = line_content[0]

    if dict.has_key(line_index):
        current_size=len(dict[line_index])
        for i in range(2,len(line_content),1):
            try:
		dict[line_index][i+current_size-2] = line_content[i]
	    except ValueError:
		continue
    else:
        dict[line_index]={}
        for i in range(1,len(line_content),1):
            try:
                dict[line_index][i] = line_content[i]
            except ValueError:
                continue

for i in dict.keys():
    print '%s\t%s\t%s\t%s\t%s' % (i,dict[i][1],dict[i][2],dict[i][3],dict[i][4])



