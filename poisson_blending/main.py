import cv2
import os
import numpy as np
import scipy.sparse as scp
from scipy.sparse.linalg import spsolve

#Main:
os.chdir(os.path.dirname(os.path.abspath(__file__)))
source = cv2.imread(os.getcwd()+'\\'+'source.jpg',cv2.IMREAD_UNCHANGED)
target = cv2.imread(os.getcwd()+'\\'+'target.jpg',cv2.IMREAD_UNCHANGED)
res = target[:,:,:]


# ,305,1005:  تصویر ساحل و مجسمه بودا
pos_i = 305
pos_j = 1005

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
    part_res = heros*helper
    helper = np.greater(heros,255)
    # up =  np.zeros((h,w))

    # cv2.normalize(heros, up, 200, 255, cv2.NORM_MINMAX)
    
    part_res  = part_res*(1-helper) + 255*helper

    res[pos_i:pos_i+h,pos_j:pos_j+w,r] = part_res[:,:]


    # print(r)

cv2.imwrite('res.jpg', res)


        











