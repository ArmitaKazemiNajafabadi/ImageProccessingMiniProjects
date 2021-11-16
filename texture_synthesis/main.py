import cv2
import os
import numpy as np
from math import sqrt


def find_matched_block_first_row(b1):
    global source, cols, rows, B, T
    sourcer = np.zeros(source.shape, dtype = 'uint8')
    sourcer[:,:,:] = source[:,:,:]
    # sourcer = cv2.cvtColor(sourcer, cv2.COLOR_BGR2HSV)
    # ss = cv2.matchTemplate(cv2.cvtColor(source[0:rows-B,0:cols-B,:],cv2.COLOR_BGR2HSV),cv2.cvtColor(b1,cv2.COLOR_BGR2HSV),cv2.TM_CCOEFF_NORMED)
    ss = cv2.matchTemplate(sourcer[0:rows-B,0:cols-B,:],b1,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(ss) 
    res = sourcer[max_loc[1]:max_loc[1]+B,max_loc[0]:max_loc[0]+B,:]
    # res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    return res

def  find_matched_block_other_row(b10,b11):
    global source, cols, rows, B, T
    sourcer = np.zeros(source.shape, dtype = 'uint8')
    sourcer[:,:,:] = source[:,:,:]
    # sourcer = cv2.cvtColor(sourcer, cv2.COLOR_BGR2HSV)

    s1 = cv2.matchTemplate(sourcer[0:rows-B,0:cols-B,:],b10,cv2.TM_CCOEFF_NORMED)
    s2 = cv2.matchTemplate(sourcer[0:rows-B,0:cols-B,:],b11,cv2.TM_CCOEFF_NORMED)
    h = min(s1.shape[0],s2.shape[0])
    w = min(s1.shape[1],s2.shape[1])
    # ss = np.sqrt(s1[0:h,0:w]**2 + s2[0:h,0:w]**2)
    ss = s1[0:h,0:w] + s2[0:h,0:w]
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(ss) 
    res = sourcer[max_loc[1]:max_loc[1]+B,max_loc[0]:max_loc[0]+B,:]
    # res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    return res

def merge_by_cut_first_row(b1,b2):
    b = cv2.cvtColor(b1-b2,cv2.COLOR_BGR2GRAY)
    dyn_dist = np.zeros(b.shape,dtype ='uint16')
    dyn_path = np.zeros(b.shape,dtype = 'uint8')
    dyn_dist[0,:] = b[0,:]
    # print(dyn_dist)
    for r in range(1,b.shape[0]):
        if dyn_dist[r-1][0] < dyn_dist[r-1][1]:
            dyn_dist[r][0] = b[r][0] + dyn_dist[r-1][0]
            dyn_path[r][0] = 0
        else:
            dyn_dist[r][0] = b[r][0] + dyn_dist[r-1][1]
            dyn_path[r][0] = 1
        # 
        for c in range(1,b.shape[1]-1):
            dyn_path[r][c] = c
            dyn_dist[r][c] = b[r][c] + min(dyn_dist[r-1][c-1],dyn_dist[r-1][c],dyn_dist[r-1][c+1])
            if dyn_dist[r][c] - b[r][c] == dyn_dist[r-1][c-1]:
                dyn_path[r][c] -= 1
            elif dyn_dist[r][c] - b[r][c] == dyn_dist[r-1][c+1]:
                dyn_path[r][c] += 1
        #    
        l = b.shape[1]-1
        dyn_path[r][l] = l
        dyn_dist[r][l] = b[r][l] + min(dyn_dist[r-1][l-1],dyn_dist[r-1][l])
        if dyn_dist[r][l] - b[r][l] == dyn_dist[r-1][l-1]:
            dyn_path[r][l] -= 1
        # 

    res = np.zeros(b1.shape, dtype = 'uint8' )
    arg = np.argmin(dyn_dist[b.shape[0]-1])
    r = b.shape[0] - 1
    c = b.shape[1] 
    while r >= 0:
        res[r,0:arg+1,:] = b1[r,0:arg+1,:]
        res[r,arg+1:c,:] = b2[r,arg+1:c,:]
        arg = dyn_path[r][arg]
        # print(arg)
        r -= 1
    return res


def merge_by_cut_first_col(b1,b2):
    b = cv2.cvtColor(b1-b2,cv2.COLOR_BGR2GRAY)
    dyn_dist = np.zeros(b.shape,dtype ='uint16')
    dyn_path = np.zeros(b.shape,dtype = 'uint8')
    dyn_dist[:,0] = b[:,0]
    # print(dyn_dist)
    for c in range(1,b.shape[1]):
        if dyn_dist[0][c-1] < dyn_dist[1][c]:
            dyn_dist[0][c] = b[0][c] + dyn_dist[0][c-1]
            dyn_path[0][c] = 0
        else:
            dyn_dist[0][c] = b[0][c] + dyn_dist[0][c-1]
            dyn_path[0][c] = 1
        # 
        for r in range(1,b.shape[0]-1):
            dyn_path[r][c] = r
            dyn_dist[r][c] = b[r][c] + min(dyn_dist[r-1][c-1],dyn_dist[r][c-1],dyn_dist[r+1][c-1])
            if dyn_dist[r][c] - b[r][c] == dyn_dist[r-1][c-1]:
                dyn_path[r][c] -= 1
            elif dyn_dist[r][c] - b[r][c] == dyn_dist[r+1][c-1]:
                dyn_path[r][c] += 1
        #    
        l = b.shape[0]-1
        dyn_path[l][c] = l
        dyn_dist[l][c] = b[l][c] + min(dyn_dist[l-1][c-1],dyn_dist[l][c-1])
        if dyn_dist[l][c] - b[l][c] == dyn_dist[l-1][c-1]:
            dyn_path[l][c] -= 1
        # 

    res = np.zeros(b1.shape, dtype = 'uint8' )
    arg = np.argmin(dyn_dist[:,b.shape[1]-1])
    r = b.shape[0] 
    c = b.shape[1] - 1
    # threshold = 50
    while c >= 0:
        res[0:arg+1,c,:] = b1[0:arg+1,c,:]
        res[arg+1:r,c,:] = b2[arg+1:r,c,:]
        # dis1 = sqrt(b1[arg][c][0]**2 + b1[arg][c][1]**2 + b1[arg][c][2]**2)
        # dis2 = sqrt(b2[arg][c][0]**2 + b2[arg][c][1]**2 + b2[arg][c][2]**2)
        # if(max(dis1,dis2)-min(dis1, dis2)>threshold):
        #     res[arg:arg+1,c,:] = (b1[arg:arg+1,c,:] + b2[arg:arg+1,c,:])//2
        arg = dyn_path[arg][c]

        # print(arg)
        c -= 1
        # res[:,:,:] = cv2.blur(res,(1,3))

    return res

def  merge_by_cut_other_col(b10,b11,b2):
    global B, T
    cut_ofoghi = merge_by_cut_first_col(b10,b2[0:T,0:B,:])
    cut_amudi = merge_by_cut_first_row(b11,b2[0:B,0:T,:])
    cut = np.zeros(b2.shape,dtype = 'uint8')
    cut[0:T,0:B,:] = cut_ofoghi
    cut[0:B,0:T,:] = cut_amudi
    return cut



#Main:
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# source = cv2.imread(os.getcwd()+'\\'+'texture1.jpg',cv2.IMREAD_UNCHANGED)
# B = 120
# T = 30

source = cv2.imread(os.getcwd()+'\\'+'texture2.jpg',cv2.IMREAD_UNCHANGED)
B = 195
T = 48

rows = source.shape[0]
cols = source.shape[1]

iter_times = (2500-B)//(B-T) + 1
# init target
target = np.zeros((iter_times*(B-T)+B+B,iter_times*(B-T)+B+B,3), dtype = 'uint8'); 
target[0:B,0:B,:] = source[100:100+B,100:100+B,:]

######### fill the first row 
x_parameter = B-T
y_parameter = 0
for i in range(1,iter_times+1):

    b1 = target[0:B,x_parameter:x_parameter+T,:]
    b2 = find_matched_block_first_row(b1)
    b3 = merge_by_cut_first_row(b1,b2[0:B,0:T,:])
    b2[0:B,0:T,:] = b3
    target[0:B,x_parameter:x_parameter+B,:] = b2
    x_parameter += B-T
# first row filled
y_parameter += B-T
x_parameter = B-T
######### fill next rows
for j in range(1,iter_times+1):
    ####### fill first column block
    x_parameter = 0
    b1 = np.zeros((T,B,3),dtype = 'uint8')
    b1 = target[y_parameter:y_parameter+T,0:B,:]
    b2 = find_matched_block_first_row(b1)
    b3 = merge_by_cut_first_col(b1,b2[0:T,0:B,:])
    b2[0:T,0:B,:] = b3[:,:,:]
    target[y_parameter:y_parameter+B,0:B,:] = b2[:,:,:]
    x_parameter += B-T
    ####### fill other columns
    for i in range(1,iter_times+1):
        b10 = np.zeros((T,B,3),dtype = 'uint8')
        b11 = np.zeros((B,T,3),dtype = 'uint8')
        b10 = target[y_parameter:y_parameter+T,x_parameter:x_parameter+B,:]
        b11 = target[y_parameter:y_parameter+B,x_parameter:x_parameter+T,:]
        b2 = find_matched_block_other_row(b10,b11)
        b3 = merge_by_cut_other_col(b10,b11,b2)
        b2[0:T,0:B,:] = b3[0:T,0:B,:]
        b2[0:B,0:T,:] = b3[0:B,0:T,:]
        target[y_parameter:y_parameter+B,x_parameter:x_parameter+B,:] = b2[:,:,:]
        x_parameter += B-T
        if(x_parameter>2500+B+B):
            break

    y_parameter += B-T
    if(y_parameter>2500+B+B):
        break



# cv2.imwrite('res01.jpg', target[0:2500,0:2500,:])
cv2.imwrite('res02.jpg', target[0:2500,0:2500,:])



