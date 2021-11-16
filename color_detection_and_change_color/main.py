#= The objectives are detecting yellow and pink objects in the given images and turning yellow flowers into red and pink flowers into blue.
# the yellow and pink colors are ditected via samples of the original images. 
# some pixels in the approximately same range of color which are not associated which the flowers should be excluded. this is also done via sampling. 

import cv2
import os
import numpy as np
def color_changer_with_given_samples(color_change, threshold, below, above, source_image, pos_samples, neg_samples):
    img = cv2.cvtColor(source_image,cv2.COLOR_BGR2RGB) #channel ordering 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # specified-color mask
    mask = cv2.inRange(img, below, above)
    # add specified-color instantces to the mask using samples
    for i in pos_samples:
        sample = cv2.imread('Sample'+str(i)+'.png') 
        sample = cv2.cvtColor(sample,cv2.COLOR_BGR2RGB) #channel ordering 
        sample = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)
        for x in np.ndindex(sample.shape[:2]):
            mask = mask | cv2.inRange(img, sample[x]-threshold,sample[x]+threshold)


    # remove unspecified-color instantces from the mask using samples
    for i in neg_samples:
            sample = cv2.imread('Sample'+str(i)+'.png') 
            sample = cv2.cvtColor(sample,cv2.COLOR_BGR2RGB) #channel ordering 
            sample = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)
            for x in np.ndindex(sample.shape[:2]):
                mask = mask & ~(cv2.inRange(img, sample[x]-[0,5,5],sample[x]+[0,5,5]))

    # extract specified-color pixels and then their channels
    spc_color = cv2.bitwise_and(img,img, mask= mask)
    h, s, v = cv2.split(spc_color)
    h1, s1, v1 = cv2.split(img)

    # color changing process
    if(color_change=='Yellow_to_Red'):
        h = h+133
        m = np.greater(h,133)   
    elif (color_change=='Pink_to_Blue'):
        h = h+25
        m = np.greater(h,25)
   
    h = h * m
    h1 = h1 + h

    # merging channels
    img = cv2.merge([h1,s1,v1])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img

threshold = [5,20,20]
threshold = [4,20,20]

os.chdir(os.path.dirname(os.path.abspath(__file__)))

cv2.imwrite('res1.jpg', color_changer_with_given_samples('Yellow_to_Red',np.array([3,10,10]), np.array([20,10,10]), np.array([24,250,250]), cv2.imread(os.getcwd()+'\\'+'Yellow.jpg'), [1,2,3], [4]))
cv2.imwrite('res2.jpg', color_changer_with_given_samples('Pink_to_Blue',np.array([5,20,20]), np.array([160,30,30]), np.array([180,250,250]), cv2.imread(os.getcwd()+'\\'+'Pink.jpg'), ['P1','P2','P3','P4','P5','P6','P7'], []))
