import cv2
import os
import numpy as np
from math import sqrt

def do_the_thing(pos_i,pos_j,r,source,target):
    pos_i_1 = pos_i//r
    pos_j_1 = pos_j//r
    source_1 = source.astype('int16')
    target_1 =  target.astype('int16')
    
    source_1_gus = cv2.GaussianBlur(source_1,(11,11),5)
    target_1_gus = cv2.GaussianBlur(target_1,(11,11),5)
    source_1_lap = source_1 - source_1_gus
    target_1_lap = target_1 - target_1_gus
 
    mask_1 =  cv2.resize(mask.astype('float32'), (mask.shape[1]//r,mask.shape[0]//r))

    mask_blur_1 =  cv2.GaussianBlur(mask_1,(15,15),5) * mask_1 

    source_1_lap_big = np.zeros(target_1_lap.shape)
    source_1_lap_big[pos_i_1:pos_i_1+source_1_lap.shape[0],pos_j_1:pos_j_1+source_1_lap.shape[1],:] = source_1_lap[:,:,:]

    blend_1 = mask_blur_1*source_1_lap_big + (1-mask_blur_1)*target_1_lap
    source_1_gus_big = np.zeros(target_1_lap.shape)
    source_1_gus_big[pos_i_1:pos_i_1+source_1_gus.shape[0],pos_j_1:pos_j_1+source_1_gus.shape[1],:] = source_1_gus[:,:,:]
    gus_blend = mask_blur_1*source_1_gus_big + (1-mask_blur_1)*target_1_gus

    return blend_1,source_1_gus,target_1_gus,gus_blend


#Main:
os.chdir(os.path.dirname(os.path.abspath(__file__)))
source = cv2.imread(os.getcwd()+'\\'+'source.jpg',cv2.IMREAD_UNCHANGED)
target = cv2.imread(os.getcwd()+'\\'+'target.jpg',cv2.IMREAD_UNCHANGED)

# ,305,1005
pos_i = 305
pos_j = 1005

h,w,d = source.shape

mask = np.zeros(target.shape, dtype='float32')


mask[pos_i:pos_i+h,pos_j:pos_j+w,:] = np.greater(source,5)

# 1_Size
blend_1,source_gus_1,target_gus_1,gus_1 = do_the_thing(pos_i,pos_j,1,source,target)
source_2 = cv2.resize(source_gus_1, (source_gus_1.shape[1]//2,source_gus_1.shape[0]//2))
target_2 =  cv2.resize(target_gus_1, (target_gus_1.shape[1]//2,target_gus_1.shape[0]//2))

# 2_Size
blend_2,source_gus_2,target_gus_2,gus_2 = do_the_thing(pos_i,pos_j,2,source_2,target_2)
source_4 = cv2.resize(source_gus_2, (source_gus_2.shape[1]//2,source_gus_2.shape[0]//2))
target_4 =  cv2.resize(target_gus_2, (target_gus_2.shape[1]//2,target_gus_2.shape[0]//2))

# 4_Size
blend_4,source_gus_4,target_gus_4,gus_4 = do_the_thing(pos_i,pos_j,4,source_4,target_4)
source_8 = cv2.resize(source_gus_4, (source_gus_4.shape[1]//2,source_gus_4.shape[0]//2))
target_8 =  cv2.resize(target_gus_4, (target_gus_4.shape[1]//2,target_gus_4.shape[0]//2))

# 8_Size
blend_8,source_gus_8,target_gus_8,gus_8 = do_the_thing(pos_i,pos_j,8,source_8,target_8)
source_16 = cv2.resize(source_gus_8, (source_gus_8.shape[1]//2,source_gus_8.shape[0]//2))
target_16 =  cv2.resize(target_gus_8, (target_gus_8.shape[1]//2,target_gus_8.shape[0]//2))

# 16_Size
blend_16,source_gus_16,target_gus_16,gus_16 = do_the_thing(pos_i,pos_j,16,source_16,target_16)

# 8_size
# helper = cv2.resize(gus_16, (gus_16.shape[1]*2,gus_16.shape[0]*2))
# res_8 = helper + cv2.resize(blend_8, (helper.shape[1],helper.shape[0]))
res_8 = cv2.resize(gus_16, (gus_16.shape[1]*2,gus_16.shape[0]*2)) + blend_8

# 4_size
# helper = cv2.resize(res_8, (res_8.shape[1]*2,res_8.shape[0]*2))
# res_4 =  helper + cv2.resize(blend_4, (helper.shape[1],helper.shape[0]))
res_4 = cv2.resize(res_8, (res_8.shape[1]*2,res_8.shape[0]*2)) + blend_4

# 2_size
# helper = cv2.resize(res_4, (res_4.shape[1]*2,res_4.shape[0]*2))
# res_2 =   helper + cv2.resize(blend_2, (helper.shape[1],helper.shape[0]))
res_2 =  cv2.resize(res_4, (res_4.shape[1]*2,res_4.shape[0]*2)) + blend_2

# 1_size
# helper = cv2.resize(res_2, (res_2.shape[1]*2,res_2.shape[0]*2))
# res_1 = helper + cv2.resize(blend_1, (helper.shape[1],helper.shape[0]))
res_1 = cv2.resize(res_2, (res_2.shape[1]*2,res_2.shape[0]*2)) + blend_1

helper = np.greater_equal(res_1,0)
res = res_1*helper
helper = np.greater(res_1,255)
res  = res*(1-helper) + 255*helper

cv2.imwrite('res.jpg',res.astype('uint8'))






