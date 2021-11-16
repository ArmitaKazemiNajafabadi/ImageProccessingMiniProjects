import cv2
import os
import numpy as np
import scipy.sparse as scp
from scipy.sparse.linalg import spsolve

#Main:
os.chdir(os.path.dirname(os.path.abspath(__file__)))
source = cv2.imread(os.getcwd()+'\\'+'2source.jpg',cv2.IMREAD_UNCHANGED)
target = cv2.imread(os.getcwd()+'\\'+'2target.jpg',cv2.IMREAD_UNCHANGED)
res = target[:,:,:]


# # ,305,1005:  تصویر ساحل و مجسمه بودا
# pos_i = 305
# pos_j = 1005

# source = cv2.resize(source,(source.shape[1]//2,source.shape[0]//2))

source = cv2.resize(source,(source.shape[1]//2,source.shape[0]//2))
target = cv2.resize(target,(target.shape[1]//2,target.shape[0]//2))

# ,1420,2450: تصویر مجسمه بودا در کوه
pos_i = 2000//2 - 90
pos_j = 1500//2 - 140

# # 465, 280: تصویر مجسمه بودا در دهانه غار
# pos_i = 410
# pos_j = 260


# target = cv2.resize(target,((target.shape[1]+1)//2,(target.shape[0]+1)//2)).astype('uint8')
# source = cv2.resize(source,(source.shape[1]//2,source.shape[0]//2)).astype('uint8')
# ,1420,2450
# pos_i = 1420//2 
# pos_j = 2450//2 


# # 1400,1129,
# pos_i = 1400//2 -10
# pos_j = 1129//2 -80

# # 1500,1050
# pos_i = 1500//2 -10
# pos_j = 1050//2 -80

# target = cv2.resize(target,((target.shape[1]+1)//2,(target.shape[0]+1)//2)).astype('uint8')
# source = cv2.resize(source,(source.shape[1]//2,source.shape[0]//2)).astype('uint8')

# # # 1150,2050
# pos_i = 1150//2 + 300
# pos_j = 2050//2 
# pos_i =  200
# pos_j = 100


h,w,d = source.shape
source_lap = cv2.Laplacian(source, cv2.CV_16S)
# threshold = 10
threshold = 50



for r in range(0,3):
    zarayeb_matrix = scp.lil_matrix((h*w,h*w))

    zarayeb_deraz = np.zeros((h*w))
    for i in range(0,h*w):
        dy = i//w
        dx = i%w
        if source[dy][dx][r] <= threshold:
            zarayeb_matrix[i,i] = 1
            zarayeb_deraz[i] = target[pos_i+dy][pos_j+dx][r] 

        elif dy + 1 >= h or dy - 1 < 0 or dx + 1 >= w or dx - 1 < 0 or source[dy+1][dx][r] <= threshold or source[dy-1][dx][r] <= threshold or source[dy][dx+1][r] <= threshold or source[dy+1][dx][r] <= threshold:
                zarayeb_matrix[i,i] = 1
                zarayeb_deraz[i] = target[pos_i+dy][pos_j+dx][r] 

        else:
            zarayeb_matrix[i,i-1] = 1
            zarayeb_matrix[i,i+1] = 1
            # zarayeb_matrix[i-w,i] = 1
            # zarayeb_matrix[i+w,i] = 1
            zarayeb_matrix[i,i-w] = 1
            zarayeb_matrix[i,i+w] = 1
            zarayeb_matrix[i,i] = -4
            zarayeb_deraz[i] = source_lap[dy][dx][r]

    zarayeb_matrix = zarayeb_matrix.tocsr()
    x = spsolve(zarayeb_matrix, zarayeb_deraz)  
    
    # cv2.normalize(x.reshape((h,w)), res, 0, 255, cv2.NORM_MINMAX)

    
    heros = np.zeros((h,w))
    heros = x.reshape((h,w))
    helper = np.greater_equal(heros,0)
    res = heros*helper
    helper = np.greater(heros,255)
    # up =  np.zeros((h,w))

    # cv2.normalize(heros, up, 200, 255, cv2.NORM_MINMAX)
    
    res  = res*(1-helper) + 255*helper

    target[pos_i:pos_i+h,pos_j:pos_j+w,r] = res[:,:]


    # print(r)

cv2.imwrite('2res.jpg', target)


        











