import cv2
import os
import numpy as np

static_img = cv2.imread('D:\Sharif Related\Terme5\Image Processing\HomeWork\MainProj1\Yellow.jpg')
sample = cv2.imread('D:\Sharif Related\Terme5\Image Processing\HomeWork\MainProj1\Sample1.png') 
sample = cv2.cvtColor(sample,cv2.COLOR_BGR2RGB) #channel ordering 
sample = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)
img = cv2.cvtColor(static_img,cv2.COLOR_BGR2RGB) #channel ordering 
img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

threshold = [3,10,10]
below_yellow = np.array([20,10,10])
above_yellow = np.array([24,250,250])

mask = cv2.inRange(img, below_yellow, above_yellow)


for x in np.ndindex(sample.shape[:2]):
    mask = mask | cv2.inRange(img, sample[x]-threshold,sample[x]+threshold)

## pos Sample
sample = cv2.imread('D:\Sharif Related\Terme5\Image Processing\HomeWork\MainProj1\Sample2.png') 
sample = cv2.cvtColor(sample,cv2.COLOR_BGR2RGB) #channel ordering 
sample = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)    

for x in np.ndindex(sample.shape[:2]):
    mask = mask | cv2.inRange(img, sample[x]-threshold,sample[x]+threshold)

## neg Sample
sample = cv2.imread('D:\Sharif Related\Terme5\Image Processing\HomeWork\MainProj1\Sample4.png') 
sample = cv2.cvtColor(sample,cv2.COLOR_BGR2RGB) #channel ordering 
sample = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)

for x in np.ndindex(sample.shape[:2]):
    mask = mask & ~(cv2.inRange(img, sample[x]-[0,5,5],sample[x]+[0,5,5]))

## pos Sample
sample = cv2.imread('D:\Sharif Related\Terme5\Image Processing\HomeWork\MainProj1\Sample3.png') 
sample = cv2.cvtColor(sample,cv2.COLOR_BGR2RGB) #channel ordering 
sample = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)    

for x in np.ndindex(sample.shape[:2]):
    mask = mask | cv2.inRange(img, sample[x]-[2,10,10],sample[x]+[2,10,10])


yellows = cv2.bitwise_and(img,img, mask= mask)
h, s, v = cv2.split(yellows)
h1, s1, v1 = cv2.split(img)


h = h+133
m = np.greater(h,133)
h = h * m
h1 = h1 + h

img = cv2.merge([h1,s1,v1])

img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR) #convert before showing

# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

os.chdir('D:\Sharif Related\Terme5\Image Processing\HomeWork\MainProj1')
# # function to save img with a given name 
cv2.imwrite('res02.jpg',img)

#######
#######
## Pink To Blue Image

static_img = cv2.imread('D:\Sharif Related\Terme5\Image Processing\HomeWork\MainProj1\Pink.jpg')
sample = cv2.imread('D:\Sharif Related\Terme5\Image Processing\HomeWork\MainProj1\SampleP1.png') 
sample = cv2.cvtColor(sample,cv2.COLOR_BGR2RGB) #channel ordering 
sample = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)
img = cv2.cvtColor(static_img,cv2.COLOR_BGR2RGB) #channel ordering 
img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

threshold = [3,10,10]
below_pink = np.array([160,30,30])
above_pink = np.array([180,250,250])

mask = cv2.inRange(img, below_pink, above_pink)
# mask = mask & 0

for x in np.ndindex(sample.shape[:2]):
    mask = mask | cv2.inRange(img, sample[x]-threshold,sample[x]+threshold)

sample = cv2.imread('D:\Sharif Related\Terme5\Image Processing\HomeWork\MainProj1\SampleP2.png') 
sample = cv2.cvtColor(sample,cv2.COLOR_BGR2RGB) #channel ordering 
sample = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)    

for x in np.ndindex(sample.shape[:2]):
    mask = mask | cv2.inRange(img, sample[x]-threshold,sample[x]+threshold)

sample = cv2.imread('D:\Sharif Related\Terme5\Image Processing\HomeWork\MainProj1\SampleP3.png') 
sample = cv2.cvtColor(sample,cv2.COLOR_BGR2RGB) #channel ordering 
sample = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)    

for x in np.ndindex(sample.shape[:2]):
    mask = mask | cv2.inRange(img, sample[x]-threshold,sample[x]+threshold)

sample = cv2.imread('D:\Sharif Related\Terme5\Image Processing\HomeWork\MainProj1\SampleP4.png') 
sample = cv2.cvtColor(sample,cv2.COLOR_BGR2RGB) #channel ordering 
sample = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)    

for x in np.ndindex(sample.shape[:2]):
    mask = mask | cv2.inRange(img, sample[x]-threshold,sample[x]+threshold)

sample = cv2.imread('D:\Sharif Related\Terme5\Image Processing\HomeWork\MainProj1\SampleP5.png') 
sample = cv2.cvtColor(sample,cv2.COLOR_BGR2RGB) #channel ordering 
sample = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)    

for x in np.ndindex(sample.shape[:2]):
    mask = mask | cv2.inRange(img, sample[x]-threshold,sample[x]+threshold)

threshold = [5,20,20]


sample = cv2.imread('D:\Sharif Related\Terme5\Image Processing\HomeWork\MainProj1\SampleP6.png') 
sample = cv2.cvtColor(sample,cv2.COLOR_BGR2RGB) #channel ordering
sample = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)    

for x in np.ndindex(sample.shape[:2]):
    mask = mask | cv2.inRange(img, sample[x]-threshold,sample[x]+threshold)

threshold = [4,20,20]

sample = cv2.imread('D:\Sharif Related\Terme5\Image Processing\HomeWork\MainProj1\SampleP7.png') 
sample = cv2.cvtColor(sample,cv2.COLOR_BGR2RGB) #channel ordering
sample = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)    

for x in np.ndindex(sample.shape[:2]):
    mask = mask | cv2.inRange(img, sample[x]-threshold,sample[x]+threshold)

# gain pinks
pinks = cv2.bitwise_and(img,img, mask= mask)

# cv2.imshow('image',pinks)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

h, s, v = cv2.split(pinks)
h1, s1, v1 = cv2.split(img)

h = h+25
m = np.greater(h,25)
h = h * m
    
h1 = h1 + h

img = cv2.merge([h1, s1 ,v1])

img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR) #convert before showing

# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

os.chdir('D:\Sharif Related\Terme5\Image Processing\HomeWork\MainProj1')
# # function to save img with a given name 
cv2.imwrite('res03.jpg',img)
