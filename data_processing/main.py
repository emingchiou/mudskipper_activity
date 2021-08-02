# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 14:24:47 2020

@author: 邱妍敏
"""
# In[initialization]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy import interpolate
from datetime import datetime
import seaborn as sns
import scipy.stats as scip
import glob, os
import ast
import math



import initiate

mud_n = initiate.mud_n
swFE = initiate.swFE # start with ebb('E') or flood('F') tide
hbftc = initiate.hbftc # hours_before_first_tidal_change
Fd = initiate.Fd # flood tide duration
Ed = initiate.Ed # ebb tide duration
Ld = initiate.Ld # light duration
Dd = initiate.Ld # dark duration

binning = initiate.binning
   
parameters = initiate.parameters    


mud_n = 'A-1'
# In[croping videos]
    
from functions import crop_img
from functions import two_roi

date = '2021-05-21-00-00'
vid_n = ['A','B','C','D','E']

for n in vid_n:
    
    crop_img(n, date)

video_name_list = sorted(glob.glob('*.avi'))

for video_name in video_name_list:
    
    two_roi(video_name, date)
    
# In[set roi]

from functions import set_roi

date = '2021-05-21-00-00'

for n in ['C-2']:    
    
    set_roi(n, date)
    

# In[import data]

from functions import data_modification

for n in mud_n:  
    
    print(n)
     
    #n = input('please enter the mud number:')
    
    # import roi
    r = pd.read_csv('./sup/mud'+n+'rois.txt',sep='\t',names=['zone','p'])   
    Water_zone = ast.literal_eval(r.iloc[0,1]) 
    Intertidal_zone = ast.literal_eval(r.iloc[1,1])
    Land_zone = ast.literal_eval(r.iloc[2,1]) 
    Wall_zone = ast.literal_eval(r.iloc[3,1]) 
    Food_zone = ast.literal_eval(r.iloc[4,1]) 
    Scale_zone = ast.literal_eval(r.iloc[5,1])
    
    # import timetag, coordination  
    xycsv = sorted(glob.glob('./mud'+n+'/*filtered.csv')) 
    timecsv = sorted(glob.glob('./mud'+n[0]+'/*.csv'))[:len(xycsv)]
    
    dfss = (data_modification(xy,time) for xy, time in zip(xycsv, timecsv))
    data = pd.concat(dfss, axis=0, ignore_index=True)
    data = data.dropna(axis=0, how='any')
    
    # data length
    datalen = len(data)
    print('data length: '+str(datalen))
    print(str(datalen/2/60/60//24)+' days and '+str(round((datalen/2/60/60%24),2))+' hrs\n')   
    data['local_timetag'] = pd.to_datetime((data.timetag+28800).astype(int), unit = 's')
    
    outlier = pd.DataFrame()
    
    # delete outlier: likelihood
    lowlike = data['like'] < 0.4
    outlier = outlier.append(data[lowlike][['timetag','local_timetag']].assign(reason = np.repeat('lowlike', len(data[lowlike]))))
    print('\nlow head likelihood\n', data[lowlike])
    data = data[~lowlike]
    
    # delete outlier: out of zone
    outzone = (data.X < Wall_zone[0][0]) | (data.X > Wall_zone[1][0]) | (data.Y < Wall_zone[0][1]) | (data.Y > Wall_zone[2][1]) 
    outlier = outlier.append(data[outzone][['timetag','local_timetag']].assign(reason = np.repeat('outzone', len(data[outzone]))))
    print('\nframes out of wall zone:\n', data[outzone])
    data = data[~outzone]
                  
    # distance  
    h = Scale_zone[1][0]-Scale_zone[0][0]
    realh = 43 #cm
    data['distance'] = np.insert((np.sqrt(np.square(np.diff(np.array(data.X))) + np.square(np.diff(np.array(data.Y)))))*(realh / h), 0,0)
    highdistance = data.distance > realh/2
    outlier = outlier.append(data[highdistance][['timetag','local_timetag']].assign(reason = np.repeat('highdistance', len(data[highdistance]))))
    print('\nframes with too high distance:\n',data[highdistance])
    data = data[~highdistance]     
  
    ## In[delete disturbed period]
    # data[~((data['local_timetag'] >= '2021-01-29 22:30') & (data['local_timetag'] < '2021-01-30 14:00'))]
  
    ## In[frame_duration]   
    f = interpolate.interp1d(np.arange(len(data)), np.array(data.timetag), fill_value="extrapolate")
    new_index = np.arange(-0.5, len(data)+0.5, 1)
    new_timetag = np.diff(f(new_index))
    data['frame_duration'] = np.array(new_timetag)
    
    # velocity & movement
    data['velocity'] = data['distance']/np.insert(np.diff(data['timetag']),0,0.5)
    data['movement'] = (data['velocity'] > 0.01)*data['frame_duration']
    
    # delete outlier: high frame duration
    high_frame_duration = data.frame_duration > (binning+1)*60
    outlier = outlier.append(data[high_frame_duration]['timetag'].to_frame().assign(reason = np.repeat('high_frame_duration', len(data[high_frame_duration]))))
    data = data[~high_frame_duration]
    
    # save outlieres
    outlier.to_csv('mud'+n+'_outlier.csv', encoding="utf-8", mode='w')
    with open ('mud'+n+'_outlier.csv','a', encoding="utf-8") as f:
        f.write(str(datalen)+'\n'+str(datalen/2/60/60//24)+' days and '+str(round((datalen/2/60/60%24),2))+' hrs\n')
        f.write('Total length: '+str(datalen))
        f.write('Outlier length: '+str(len(outlier)))
        f.write('Frame excluded ratio: '+str(len(outlier)/datalen*100)+'%')
    print('Frame excluded ratio: '+str(round(len(outlier)/datalen*100,2))+'%')    
    
    # slicing data
    data1_tidal = data[(data['local_timetag'] >= '2021-04-28 09:00') & (data['local_timetag'] < '2021-05-12 9:00')]
    data1_notidal = data[(data['local_timetag'] >= '2021-05-12 09:00') & (data['local_timetag'] < '2021-05-26 21:00')]
    data1_all = data
    
    for data1, section in zip([data1_all,data1_tidal,data1_all],['all','tidal','notidal']):
        data1.to_csv('mud'+n+'_data1_'+section+'.csv', encoding="utf-8", mode='w')


# In[time computation]
    
for n in mud_n:
    
    print(n)
  
    data = pd.read_csv('mud'+n+'_data1_all.csv',index_col=0).copy()
    data['local_timetag'] = pd.to_datetime(data['local_timetag'])
    
    data = data.drop(columns = ['like','tail_X','tail_Y','tail_like'])    
    
    #n = input('please enter the mud number:')
    r = pd.read_csv('./sup/mud'+n+'rois.txt',sep='\t',names=['zone','p'])
    
    Water_zone = ast.literal_eval(r.iloc[0,1]) 
    Intertidal_zone = ast.literal_eval(r.iloc[1,1])
    Land_zone = ast.literal_eval(r.iloc[2,1]) 
    Wall_zone = ast.literal_eval(r.iloc[3,1]) 
    Food_zone = ast.literal_eval(r.iloc[4,1]) 
    Scale_zone = ast.literal_eval(r.iloc[5,1])

    zones_xy = [Water_zone, Intertidal_zone, Land_zone, Wall_zone, Food_zone] 
    zones = ['Water_zone', 'Intertidal_zone', 'Land_zone', 'Wall_zone', 'Food_zone']
    
    for zone, zone_name in zip(zones_xy, zones):
        zone_x1 = zone[0][0]
        zone_x2 = zone[1][0]
        zone_y1 = zone[0][1]
        zone_y2 = zone[2][1]
        
        inzone = (zone_x1 < data.X) & (data.X < zone_x2) & (zone_y1 < data.Y) & (data.Y < zone_y2)
        data[zone_name] = np.where(inzone, data['frame_duration'], 0)
    
    
    in3zone = (data.Water_zone > 0) | (data.Intertidal_zone > 0) | (data.Land_zone > 0)
    data['Wall_zone'] = np.where(in3zone, 0, data['Wall_zone'])
    
    data2 = data.iloc[:,3:].groupby(pd.Grouper(key='local_timetag', freq=str(binning)+'min')).sum()      
    data2['All_zone'] = data2['Water_zone'] + data2['Intertidal_zone'] + data2['Land_zone'] + data2['Wall_zone']
    #Totalframeduration = np.sort(Data.Water_zone+Data.Intertidal_zone+Data.Land_zone+Data.Wall_zone)[::-1]
    
    #blankatbegining = pd.DataFrame(0, index=list(pd.date_range(start='2020-09-23 09:00:00', end='2020-09-23 09:30:00', freq='10min')), columns=['distance']).reset_index().rename({'index': 'timetag'}, axis='columns')
    #data2 = data2.reset_index().drop(columns = ['frame_duration'])
    data2 = data2.reset_index()
        
    #label H-L tide or L-D light
    bph = int(60/binning)
    one_tidal_cycle = np.append(np.repeat([0], Ed*bph), np.repeat([1], Fd*bph)) # E=0, F=1
    one_LD_cycle = np.append(np.repeat([0], Ld*bph), np.repeat([1], Dd*bph)) # L=0, D=1
    initial_tide = np.ones(int(6*hbftc)) if swFE == 'F' else np.append(np.zeros(int(6*hbftc)), np.repeat([1], Fd*bph))  # tides_before_first_tidal_change
    data2['tide_level'] = np.insert(np.tile(one_tidal_cycle, (len(data2)//len(one_tidal_cycle))+1), 0, initial_tide)[:len(data2)]
    data2['light'] = np.tile(one_LD_cycle, (len(data2)//len(one_LD_cycle))+1)[:len(data2)]
    
    
    data2['Wet_zone'] = np.where(data2['tide_level'] == 0, data2['Water_zone'], data2['Water_zone']+data2['Intertidal_zone'])
    data2['Dry_zone'] = np.where(data2['tide_level'] == 1, data2['Land_zone'], data2['Land_zone']+data2['Intertidal_zone'])
    
    data2['tide_level'] = np.where((data2['local_timetag'] >= '2021-05-12 9:00'),1,data2['tide_level'])
    data2['light'] = np.where((data2['local_timetag'] >= '2021-05-12 9:00'),1,data2['light'])
    
    data2_tidal = data2[(data2['local_timetag'] >= '2021-05-05 9:00') & (data2['local_timetag'] < '2021-05-12 9:00')] #6.5days, 2/3 9:00 to 2/9 21:00
    data2_notidal = data2[(data2['local_timetag'] >= '2021-05-12 9:00') & (data2['local_timetag'] < '2021-05-19 9:00')] #7.5days, 2/9 21:00 to 2/17 9:00
    # data2_longtidal = data2[(data2['local_timetag'] >= '2021-01-27 21:00') & (data2['local_timetag'] < '2021-02-09 21:00')] #7.5days, 2/9 21:00 to 2/17 9:00
    # data2_trans = data2[(data2['local_timetag'] >= '2021-02-03 09:00') & (data2['local_timetag'] < '2021-02-17 09:00')]
    data2_all = data2
    
    
    for data2, tidal in zip([data2_tidal, data2_all, data2_notidal],['tidal','all','notidal']):
        data2.to_csv('mud'+n+'_data2_'+tidal+'.csv', encoding="utf-8")
        
# In[double plot]          

from functions import doubleplot

for n in ['A-1','A-2']:   
    print(n)    
    for section in ['notidal']:
        for parameter in parameters:
            
            doubleplot(n, section, parameter)
            
# In[Chi-Square]
    
from functions import chisquare

for n in mud_n:
    
    print(n)
        
    with open ('mud'+n+'_chi.csv', 'w') as f:
        f.write('parameter,section,Tau,Amplitude,Amplitude-p\n' )
        
    for section in ['tidal','notidal']:     
        
        for parameter in parameters:
            #chi_target = input('Which data are you going to analyze?')
            chisquare(n, section, parameter)

# In[poolchi]

from functions import poolchi

poolchi(mud_n)

# In[poolsummary]

from functions import summary

sections = ['']
summary(mud_n, sections, parameters)  
                
# In[scatter plot] not yet finish

for n in mud_n:
    
    data = pd.read_csv('mud'+n+'_todate_data.csv').copy()
    distribution = data[['X', 'Y', 'local_timetag']]
    distribution['Y'] = -distribution['Y']
    
    pos_L = distribution.set_index('local_timetag').between_time('9:00:00', '20:59:00')
    pos_D = distribution.set_index('local_timetag').between_time('21:00:00', '8:59:00')
    pos_L = pos_L[::30]
    pos_D = pos_D[::30]
    
    pos_F = distribution.filter(distribution.timetag)####
    pos_E = distribution.set_index('timetag')#####
    pos_F = pos_F[::30]
    pos_E = pos_E[::30]
    
    scatter=data.iloc[::120]    
    plt.scatter(scatter['X'], -scatter['Y'],alpha=0.1)
    #plt.scatter(scatter['X'], -scatter['Y'])                                      
    plt.title('Staying preference', fontsize=15) 
    plt.xticks(np.arange(0,))                                          
    plt.yticks(np.arange(-20, -100, 10),['0','5','10','15','20','25'])
    plt.xticks(np.arange(70, 550, 52),['0','5','10','15','20','25','30','35','40','45'])  
    plt.xlabel('X (cm)')                                          
    plt.ylabel('Y (cm)')                                         
    plt.show()

# In[scatter plot] not yet finish

for n in mud_n:
    
    data = pd.read_csv('mud'+n+'_todate_data.csv').copy()
    distribution = data[['X', 'Y', 'local_timetag']]
    distribution['Y'] = -distribution['Y']
    
    
    pos_F = distribution.filter(distribution.timetag)####
    pos_E = distribution.set_index('timetag')#####
    pos_F = pos_F[::30]
    pos_E = pos_E[::30]
    
    scatter=data.iloc[::120]    
    plt.scatter(scatter['X'], -scatter['Y'],alpha=0.1)
    #plt.scatter(scatter['X'], -scatter['Y'])                                      
    plt.title('Staying preference', fontsize=15) 
    plt.xticks(np.arange(0,))                                          
    plt.yticks(np.arange(-20, -100, 10),['0','5','10','15','20','25'])
    plt.xticks(np.arange(70, 550, 52),['0','5','10','15','20','25','30','35','40','45'])  
    plt.xlabel('X (cm)')                                          
    plt.ylabel('Y (cm)')                                         
    plt.show()

# In[histogram]       

import scipy.stats as scip
import seaborn as sns

#sns.distplot(data2.distance)
#plt.show()

# sns.distplot(round(data['distance'],2), bins=2000)
# plt.xticks(np.arange(min(data['distance']), max(data['distance'])+1, 0.01))
# plt.xlim(0.01,0.15)
# plt.show()

sns.distplot(data['velocity'],bins=5000)
#plt.xticks(np.arange(min(data['distance']), max(data['distance'])+1, 0.01))
plt.xlim(0,0.2)
plt.show()

# sns.distplot(data['velocity'],bins=5000)
# plt.xticks(np.arange(min(data['distance']), max(data['distance'])+1, 0.02))
# plt.xlim(0,0.2)
# plt.show()

# sns.distplot(np.sort(data['velocity']*2-data['distance']))

#%%


            
#%%        
        #boxplot
        distance_L = L['distance']/100
        distance_D = D['distance']/100
        distance_L = distance_L[distance_L>0]
        distance_D = distance_D[distance_D>0]
        
        distance_F = F['distance']/100
        distance_E = E['distance']/100
        distance_F = distance_F[distance_F>0]
        distance_E = distance_E[distance_E>0]
        
        plt.title('Activities ', fontsize=15)
        plt.ylabel('Travel distance (m/12hr)')
        plt.boxplot([distance_L, distance_D],labels=['Day','Night'])
        #plt.boxplot([distance_F, distance_E],labels=['High tide','Low tide'])
        plt.show()
        
        
        #barplot
        time_L = L[L.columns.difference(['distance'])]
        time_D = D[D.columns.difference(['distance'])]
        #time_L = time_L[time_L>0]
        #time_D = time_D[time_D>0]
        
        time_F = F[L.columns.difference(['distance'])]
        time_E = E[D.columns.difference(['distance'])]
        #time_F = time_L[time_L>0]
        #time_E = time_D[time_D>0]
        
        time_L = L[['Water_zone','Land_zone','Wall_zone']].sum()
        time_D = D[['Water_zone','Land_zone','Wall_zone']].sum()
        time_L = time_L/time_L.sum()
        time_D = time_D/time_D.sum()
        
        plt.title('Staying zone in day and at night')
        plt.margins(y=0.6)
        height = 0.4
        plt.barh(['Day','Night'],[time_L[0],time_D[0]], label='Water', height=height, color='#5599FF', edgecolor='white')
        plt.barh(['Day','Night'],[time_L[1],time_D[1]], left=[time_L[0],time_D[0]], label='Land', height=height, color='#BB5500', edgecolor='white')
        plt.barh(['Day','Night'],[time_L[2],time_D[2]], left=[time_L[0]+time_L[1], time_D[0]+time_D[1]], label='Wall', height=height, color='#BBBB00', edgecolor='white')
        plt.legend(bbox_to_anchor=(1,1), ncol=1)
        plt.xticks(np.arange(0, 1.01, 0.2),['0','20','40','60','80','100'])
        plt.xlabel('time (%)')
        #plt.xticks([0,1],['Day','Night'])
        plt.show()
