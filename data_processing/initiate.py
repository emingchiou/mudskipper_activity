# -*- coding: utf-8 -*-
"""
Created on Sun May 16 16:11:36 2021

@author: 邱妍敏
"""

mud_n = ['A-1','A-2','B-1','B-2','C-1','C-2','D-1','D-2','E-1','E-2']
#mud_n = ['A-1','A-2','B-1','B-2','C-1','C-2','D-1','D-2']

swFE = 'F' # start with ebb('E') or flood('F') tide
#swFE = 'E' # start with ebb('E') or flood('F') tide

hbftc = 4.5 # hours_before_first_tidal_change
#hbftc = 4 # hours_before_first_tidal_change

Fd = 6.5 # flood tide duration
Ed = 6.5 # ebb tide duration
Ld = 12 # light duration
Dd = 12 # dark duration

binning = 10  
   
parameters = ['distance', 'movement', \
              'Water_zone', 'Intertidal_zone',\
              'Land_zone', 'Wall_zone', 'Food_zone', \
              'Wet_zone','Dry_zone','All_zone']

    
    