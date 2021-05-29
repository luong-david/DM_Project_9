# -*- coding: utf-8 -*-
"""
Created on Fri May 28 20:29:50 2021

@author: e399410
"""


def checkKey(dict, key):
      
    if key in dict:
        return True
    else:
        return False
    
def getAttributesList(dict):
    att_set = set()
    for item in dict:
        if item['attributes'] is not None:
            for att in list(item['attributes'].keys()):
                att_set.add(att)
    return list(att_set)    