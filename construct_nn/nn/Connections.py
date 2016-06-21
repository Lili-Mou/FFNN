# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 14:40:33 2014

@author: mou
"""

import numpy as np
class connection:
    con_type = 'l'
    xlayer = None
    ylayer = None
    xnum = None
    ynum = None
    Widx = None # reshape to lnum by unum
    Wcoef = None
    ydownid = 0
    def __init__(self, xlayer, ylayer, Widx, Wcoef=1.0):
        self.xlayer = xlayer
        self.ylayer = ylayer
        self.Widx = Widx
        self.Wcoef = Wcoef
        xlayer.connectUp.append(self)
        ylayer.connectDown.append(self)
        			
class BilinearConnection():
    con_type = 'b'
    xlayer1 = None
    xlayer2 = None
    ylayer  = None
    xnum = None
    ynum = None
    Widx = -1
    Wcoef = 1.0
    ydownid = 0
    def __init__(self, xlayer1, xlayer2, ylayer, Widx):
        self.xlayer1 = xlayer1
        self.xlayer2 = xlayer2
        self.ylayer  = ylayer
        self.Widx     = Widx
        self.ylayer.connectDown.append(self)
        
class PoolConnection():
    con_type = 'p'
    xlayer = None
    ylayer = None
    xnum = None
    ynum = None
    ydownid = 0
    def __init__(self, xlayer, ylayer):
        self.xlayer = xlayer
        self.ylayer = ylayer
        xlayer.connectUp.append(self)
        ylayer.connectDown.append(self)

