# -*- coding: utf-8 -*-
"""
Created on Sun May 16 03:04:36 2021

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
#%%
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

# In[crop_img]

import numpy as np
import pandas as pd
import cv2

#vid_n = ['A','B','C','D','E']


def crop_img(n, date):
    
    video_name = './mud'+n+'/mud'+n+str(date)
    video = cv2.VideoCapture(video_name +'.avi')
    
    success, frame = video.read()
    c = 0
    while success:
        success, frame = video.read()
        c += 1
        if c == 600:
            cv2.imwrite('./sup/'+video_name+'_screen shot.png', frame)
            video.release()
            break
            
    #cv2.destroyAllWindows()   
    
    img = cv2.imread(video_name+'_screen shot.png')
  
    ROIs =[]
    def ROI_selection(img):
        im = img
        r = cv2.selectROI('select', img) #roi
        x, y, w, h = r
        ROIs.append(r)
    
        img_2 = cv2.rectangle(img=im, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
        cv2.imshow('select', im)
        return img_2 
        cv2.destroyWindow('select')
    
    for i in range(0,2): 
        ROI_selection(img)
    
    roi1 = ROIs[0]
    roi2 = ROIs[1]
    
    with open ('./sup/'+video_name[7:]+'_roi.txt','w') as f:
        f.write(str(roi1))
        f.write('\t')
        f.write(str(roi2))

    cv2.destroyAllWindows()    

def test():
    a = 1
    return a

# In[2roi]

import numpy as np
import pandas as pd
import cv2
import tqdm
import time
import glob, os
import ast

#video_name_list = sorted(glob.glob('*.avi'))  
   
def two_roi(video_name, date):

    with open ('./sup/mud'+video_name[3]+date+'_roi.txt', 'r' ) as f:
        roi = f.readline().split('\t')
        roi1 = ast.literal_eval(roi[0])   
        roi2 = ast.literal_eval(roi[1])         
     
    video = cv2.VideoCapture(video_name)
    
    fps = video.get(cv2.CAP_PROP_FPS)
    frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    size1 = (roi1[2],roi1[3])
    size2 = (roi2[2],roi2[3])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    crop1 = cv2.VideoWriter(video_name[:4]+'-1_'+video_name[4:], fourcc, 2.0, size1)  
    crop2 = cv2.VideoWriter(video_name[:4]+'-2_'+video_name[4:], fourcc, 2.0, size2)  
    
    video = cv2.VideoCapture(video_name)
    
    bar = tqdm.tqdm(total=frameCount, desc=str(video_name))
    framec = 1
    success, frame = video.read()
    try:
        while success:    
            success, frame = video.read()
            # y: y+h, x: x+w
            crop_img1 = frame[roi1[1]:roi1[1]+roi1[3], roi1[0]:roi1[0]+roi1[2]]
            crop_img2 = frame[roi2[1]:roi2[1]+roi2[3], roi2[0]:roi2[0]+roi2[2]]               
            crop1.write(crop_img1)
            crop2.write(crop_img2)       
                
            cv2.imshow('crop1', crop_img1)
            cv2.imshow('crop2', crop_img2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            bar.update(1)
            framec += 1
            if framec == frameCount:
                crop1.release()
                crop2.release()
                bar.close()
                break
    except:       
        crop1.release()
        crop2.release()
        bar.close()
        raise
        
    print('\nfinish writing')


# In[set_roi]

import cv2


 # exact ROI points

def set_roi(n, date):
        
    video = cv2.VideoCapture('./mud'+n+'/mud'+n+'_'+str(date)+'.avi')
    
    success, frame = video.read()
    c = 0
    while success:
        success, frame = video.read()
        c += 1
        if c == 600:
            cv2.imwrite('./sup/'+n+'_screen shot.png', frame)
            video.release()
            break
            
    #cv2.destroyAllWindows()       
    
    img = cv2.imread('./sup/'+n+'_screen shot.png')
    img2 = img
    ROIs = []
    
    def ROI_selection(img):
        im = img
        r = cv2.selectROI('select', img) #roi
        # r = (x,y,w,h) 分别代表矩形左上角座標 (x, y) 與矩形宽度 w 跟高度 h
        # imCrop = img[y : y+h, x:x+w]
        left_upper = [r[0], r[1]]
        right_upper = [r[0] + r[2], r[1]]
        left_lower = [r[0],r[1] + r[3]]
        right_lower =  [r[0] + r[2], r[1] + r[3]]
    
        ROI_point = [left_upper, right_upper, left_lower, right_lower]
        ROIs.append(ROI_point)
        
        x, y, w, h = r
        img_2 = cv2.rectangle(img=im, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
        cv2.imshow('select', im)
        return img_2
        #cv2.destroyWindow('select')
        
    for i in range(0,6): #define ROI number (High tide)
        ROI_selection(img)
        
    Water_zone = ROIs[0]
    Intertidal_zone = ROIs[1]
    Land_zone = ROIs[2]
    Wall_zone = ROIs[3]
    Food_zone = ROIs[4]
    Scale_zone = ROIs[5]
        
    cv2.destroyAllWindows()
      
    with open ('./sup/mud'+n+'rois.txt','a') as f:
        f.write('Water_zone,\t'+str(Water_zone))
        f.write('\nIntertidal_zone,\t'+str(Intertidal_zone))
        f.write('\nLand_zone,\t'+str(Land_zone))
        f.write('\nWall_zone,\t'+str(Wall_zone))
        f.write('\nFood_zone,\t'+str(Food_zone))
        f.write('\nScale_zone,\t'+str(Scale_zone))

# In[data modification]
        
def data_modification(xy,time):
    df1 = pd.read_csv(xy).copy()
    data_xy = pd.DataFrame(df1.iloc[2:,1:]).astype(float).dropna()
    data_xy.columns=['X','Y','like','tail_X','tail_Y','tail_like']
    df2 = pd.read_csv(time).copy()
    data_timetag = pd.DataFrame(df2).astype(float)
    data_timetag.columns=['timetag']
    data5 = pd.concat([data_xy,data_timetag], axis = 1).dropna(axis=0, how='any')
    return data5

# In[doubleplot]
        
def doubleplot(n, section, plot):
    
    if section != 'trans':
        data2 = pd.read_csv('mud'+n+'_data2_'+section+'.csv',index_col=0).copy()
    else:
        data_tidal = pd.read_csv('mud'+n+'_data2_tidal.csv',index_col=0).copy()
        data_notidal = pd.read_csv('mud'+n+'_data2_notidal.csv',index_col=0).copy()
        data2 = pd.concat([data_tidal,data_notidal])
    
    
    bph = 60/binning # bins per hour
    bpd = int((Ld+Dd)*bph)
    rows = int(len(data2)//bpd)-1
    
    title = plot.capitalize() \
    if plot == 'distance' or plot == 'movement' or plot == 'velocity' \
    else 'Time in '+str(plot.replace('_', ' ').lower())
    
    fig = plt.figure()   
    gs1 = fig.add_gridspec(rows, 1, hspace=0, wspace=0)
    ax = gs1.subplots(sharex=True, sharey=True)
    fig.suptitle(title, fontsize=20)
    plt.xticks([0, Ld*bph, (Ld+Dd)*bph, (Ld+Dd+Ld)*bph, (Ld+Dd+Ld+Dd)*bph],\
                ['ZT0','ZT'+str(Ld),'ZT'+str(Ld+Dd)+'/0','ZT'+str(Ld),'ZT'+str(Ld+Dd)], fontsize=12) 
    
    
    for d in range(rows):
        x = np.arange(bpd*2)
        y = data2[plot][d*bpd:(d+2)*bpd].values.squeeze()
        zero = pd.DataFrame(0, index=[-1], columns=data2.columns)
        p = pd.concat([zero, data2[d*bpd:(d+2)*bpd],zero]).reset_index()
        floods = p[p['tide_level'].diff() == 1].index.to_list()
        ebbs = p[p['tide_level'].diff() == -1].index.to_list()
        lights = p[p['light'].diff() == 1].index.to_list()
        darks =p[p['light'].diff() == -1].index.to_list()
        
        ax[d].bar(x, y, color='black', width = 1)
        for i in range(len(floods)):
            ax[d].axvspan(floods[i], ebbs[i], facecolor='#7cc9ff', alpha=0.4)
        for i in range(len(lights)):
            ax[d].axvspan(darks[i], lights[i], facecolor='#8E8E8E', alpha=0.4)
        ax[d].yaxis.set_visible(False)
        ax[d].spines['top'].set_visible(False)
        ax[d].spines['right'].set_visible(False)
        ax[d].spines['bottom'].set_visible(False)
        ax[d].spines['left'].set_visible(False)
        if d == 0: 
            ax[d].yaxis.set_visible(True)
    
    fig.savefig(n+'_'+plot+'_'+section+'.png',encoding="utf-8")
    fig.show()


# In[chisquare]
    
def chisquare(n, section, plot):
 
    data2 = pd.read_csv('mud'+n+'_data2_'+section+'.csv',index_col=0).copy()
    
    data_chi = data2[plot]
    
    bph = 60/binning
    title = plot.capitalize() \
    if plot == 'distance' or plot == 'movement' or plot == 'velocity' \
    else 'Time in '+str(plot.replace('_', ' ').lower())
    
    N = len(data_chi)
    M = np.mean(data_chi)
    
    LowSum = sum(list(np.power(data_chi-M,2)))
    
    def Qp(s):
        P = s # period
        K = round(N/P) # the number of rows (‘days’) in P columns
        Mhsum = 0
        j = 0
        while j < P:  
            #Mh = np.power(np.mean(data_chi[j::P]) - M,2)
            Mh = (np.mean(data_chi[j:int(N-K):P]) - M) ** 2
            Mhsum += Mh
            j += 1       
        return K * N * Mhsum / LowSum
    
    period = np.arange(1, int(35*bph))
    xaxis = period/bph #(hr)
    Qplist = list(map(lambda p : Qp(p), period))
    chivalue = list(scip.chi2.ppf(0.95, period-1)) #degree of freedom = Period -1   
    
    plt.plot(xaxis, Qplist, color='black')
    plt.plot(xaxis, chivalue, color='r')
    plt.xlabel('Period (hr)',fontsize=12)
    plt.ylabel('Amplitude',fontsize=12)
    plt.title(title, fontsize=20)
    plt.savefig(n+'_'+plot+'_'+section+'_chi.png')
    plt.show()           
    
    for y, p, x in zip(Qplist, chivalue, xaxis):
        with open ('mud'+n+'_chi.csv', 'a') as f:
            f.write(plot+','+section+','
            +str(np.around(x,decimals=2))+','
            +str(np.around(y,decimals=2))+','
            +str(np.around(y-p,decimals=2))+'\n')


# In[poolchi]           


# binning = 10

def poolchi(mud_n):
        
    pool = pd.DataFrame()

    for n in mud_n:    
        pool[n] = pd.read_csv('mud'+n+'_chi.csv').set_index(['section','parameter'])['Amplitude']
    
    
    pool['mean'] = pool.mean(axis=1)
    pool['std'] = pool.iloc[:,:-2].std(axis=1)
    pool.to_csv('pool_chi.csv')
    data_pool = list(pool[['mean','std']].groupby(['section','parameter']))
    
    y = [np.array(data_pool[n][1]['mean']) for n in np.arange(0, len(data_pool))]
    y_err = [np.array(data_pool[n][1]['std']) for n in np.arange(0, len(data_pool))]
    
    bph = 60/binning     
    period = np.arange(1, int(35*bph))
    xaxis = period/bph #(hr)
    chivalue = list(scip.chi2.ppf(0.95, period-1)) #degree of freedom = Period -1    
     
    
    for i in np.arange(0, len(data_pool)):
        plt.plot(xaxis, y[i], color='black')
        plt.plot(xaxis, chivalue, color='r')
        plt.title(data_pool[i][0][1].replace('_', ' ').capitalize())
        plt.fill_between(xaxis, y[i]-y_err[i], y[i]+y_err[i], alpha=0.2)
        plt.xlabel('Period (hr)',fontsize=11)
        plt.ylabel('Amplitude',fontsize=11)
        plt.savefig('pool_'+data_pool[i][0][1]+'_'+data_pool[i][0][0]+'_chi.png')
        plt.show()

        
# In[stastistic]  
        
def summary(mud_n, sections, parameters):
    
    for section in sections:
        pool = pd.DataFrame()       
        for n in mud_n:
            data2 = pd.read_csv('mud'+n+'_data2_'+section+'.csv',index_col=0).copy()  
            
            LD = data2.groupby(['light']).mean()
            FE = data2.groupby(['tide_level']).mean()
            LDFE = data2.groupby(['light','tide_level']).mean().rename(index={1:11,0:10})
            merge = pd.concat([LD,FE,LDFE]).assign(subject = n)
            pool = pd.concat([pool, merge])
           
        pool = pool.set_index([pool.index,'subject']).sort_index()
        pool = pool.rename(index={11: 'F', 10: 'E', 1: 'D', 0: 'L', (0,0): 'LE', (0,1): 'LF', (1,0): 'DE', (1,1): 'DF'}).T
        pool.to_csv('pool_summary_'+section+'.csv') 

