import cv2
import os
import numpy as np 
import scipy.signal 


deerPoints = list()
lionPoints = list()

def mouse_listener_deer(event, x, y, flags, param):
   global deerPoints
   if event == cv2.EVENT_LBUTTONDOWN:
       deerPoints.append(np.array([x,y]))

def mouse_listener_lion(event, x, y, flags, param):
   global lionPoints
   if event == cv2.EVENT_LBUTTONDOWN:
       lionPoints.append(np.array([x,y]))

def generate_guassian(x,y,sigma):
    xd = scipy.signal.gaussian(x, sigma).reshape(x, 1)
    yd = scipy.signal.gaussian(y, sigma).reshape(y, 1)
    res = np.outer(xd, yd)
    return res


def get_cutoff(x,y,r):
    res = np.zeros((x,y))
    xx = (x+1)//2
    yy = (y+1)//2
    for i in range(xx-r,xx+r):
        for j in range(yy-r,yy+r):
            if ((i-xx)**2 + (j-yy)**2) <= r*r:
                res[i][j] = 1
    return res


os.chdir(os.path.dirname(os.path.abspath(__file__)))
lion = cv2.imread( os.getcwd()+'\\'+'q4_02_far.jpg',cv2.IMREAD_UNCHANGED)
deer = cv2.imread( os.getcwd()+'\\'+'q4_01_near.jpg',cv2.IMREAD_UNCHANGED)
# eshtebah 
img = deer
deer = lion
lion = img

cv2.namedWindow("deer_win")
cv2.setMouseCallback("deer_win", mouse_listener_deer)
cv2.imshow("deer_win",deer)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.namedWindow("lion_win")
cv2.setMouseCallback("lion_win", mouse_listener_lion)
cv2.imshow("lion_win",lion)
cv2.waitKey(0)
cv2.destroyAllWindows()

#transform deer image
src = np.array([deerPoints[0], deerPoints[1] ,deerPoints[2]]).astype(np.float32)
dst = np.array([deerPoints[0]/2+lionPoints[0]/2, deerPoints[1]/2+lionPoints[1]/2 ,deerPoints[2]/2+lionPoints[2]/2]).astype(np.float32)

matrix = cv2.getAffineTransform(src, dst)
warp_deer_mid = cv2.warpAffine(deer, matrix, (int((lion.shape[1]+deer.shape[1])/2), int((lion.shape[0]+deer.shape[0])/2)))
#transform lion image
src = np.array([lionPoints[0], lionPoints[1] ,lionPoints[2]]).astype(np.float32)

matrix = cv2.getAffineTransform(src, dst)
warp_lion_mid = cv2.warpAffine(lion, matrix, (int((lion.shape[1]+deer.shape[1])/2), int((lion.shape[0]+deer.shape[0])/2)))

# crop around images
warp_lion = np.zeros((warp_lion_mid.shape[0]-180,warp_lion_mid.shape[1]-380,3))
warp_lion[:,:,:] = warp_lion_mid[10:warp_lion_mid.shape[0]-170,190:warp_lion_mid.shape[1]-190,:]
warp_deer = np.zeros((warp_deer_mid.shape[0]-180,warp_deer_mid.shape[1]-380,3))
warp_deer[:,:,:] = warp_deer_mid[10:warp_deer_mid.shape[0]-170,190:warp_deer_mid.shape[1]-190,:]
cv2.imwrite('q4_03_far.jpg', warp_deer)
cv2.imwrite('q4_04_near.jpg', warp_lion)
# save transformed images
# cv2.imwrite('q4_03_near.jpg', warp_deer_mid)
# cv2.imwrite('q4_04_far.jpg', warp_lion_mid)

# warp_deer = cv2.imread( os.getcwd()+'\\'+'q4_03_near.jpg',cv2.IMREAD_UNCHANGED)
# warp_lion = cv2.imread( os.getcwd()+'\\'+'q4_04_far.jpg',cv2.IMREAD_UNCHANGED)

#frequency domain
b, g, r = cv2.split(warp_deer)
deer_fft_b = np.fft.fft2(b)
deer_fft_g = np.fft.fft2(g)
deer_fft_r = np.fft.fft2(r)
shifted_deer_b = np.fft.fftshift(deer_fft_b)
shifted_deer_g = np.fft.fftshift(deer_fft_g)
shifted_deer_r = np.fft.fftshift(deer_fft_r)
shifted_deer = np.zeros((shifted_deer_b.shape[0],shifted_deer_b.shape[1],3), dtype=type(shifted_deer_b[0][0]))

shifted_deer[:,:,0] = shifted_deer_b[:,:]
shifted_deer[:,:,1] = shifted_deer_g[:,:]
shifted_deer[:,:,2] = shifted_deer_r[:,:]


b, g, r = cv2.split(warp_lion)
lion_fft_b = np.fft.fft2(b)
lion_fft_g = np.fft.fft2(g)
lion_fft_r = np.fft.fft2(r)
shifted_lion_b = np.fft.fftshift(lion_fft_b) 
shifted_lion_g = np.fft.fftshift(lion_fft_g)
shifted_lion_r = np.fft.fftshift(lion_fft_r)
shifted_lion = np.zeros((shifted_lion_b.shape[0],shifted_lion_b.shape[1],3), dtype=type(shifted_lion_b[0][0]))
shifted_lion[:,:,0] = shifted_lion_b[:,:]
shifted_lion[:,:,1] = shifted_lion_g[:,:]
shifted_lion[:,:,2] = shifted_lion_r[:,:]

amp_deer = (np.abs(shifted_deer))
amp_lion = (np.abs(shifted_lion))


log_amp_deer = np.log(amp_deer)
log_amp_lion = np.log(amp_lion)

log_amp_deer_show = np.zeros(log_amp_deer.shape, dtype='uint8')
log_amp_lion_show = np.zeros(log_amp_lion.shape, dtype='uint8')

cv2.normalize(log_amp_deer,log_amp_deer_show,0,255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
cv2.normalize(log_amp_lion,log_amp_lion_show,0,255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)

cv2.imwrite('q4_05_far.jpg',log_amp_deer_show)
cv2.imwrite('q4_06_near.jpg',log_amp_lion_show)

r = 20

s = 15

x = amp_deer.shape[0]
y = amp_deer.shape[1]
if(x%2==0):
    x=x+1
if(y%2==0):
    y=y+1

low_pass_filter = generate_guassian(x,y,s)
low_pass_filter_show = low_pass_filter*(255/np.max(low_pass_filter))


x = amp_lion.shape[0]
y = amp_lion.shape[1]

if(x%2==0):
    x=x+1
if(y%2==0):
    y=y+1

high_pass_filter = 1-generate_guassian(x,y,r)
high_pass_filter_show = high_pass_filter*(255/np.max(high_pass_filter))


cv2.imwrite('q4_08_lowpass_'+str(s)+'.jpg', low_pass_filter_show)
cv2.imwrite('q4_07_highpass_'+str(r)+'.jpg', high_pass_filter_show)

cutoff_high_radi = 8
cutoff_low_radi = 12

cutoff_high_filter = 1-get_cutoff(x,y,int(cutoff_high_radi))
cutoff_low_filter = get_cutoff(x,y,int(cutoff_low_radi))

lowpass_cutoff_show = np.multiply(cutoff_low_filter,low_pass_filter_show)
highpass_cutoff_show = np.multiply(cutoff_high_filter,high_pass_filter_show)
lowpass_cutoff = np.multiply(cutoff_low_filter,low_pass_filter)
highpass_cutoff= np.multiply(cutoff_high_filter,high_pass_filter)
cv2.imwrite('q4_10_lowpass_cutoff.jpg', lowpass_cutoff_show)
cv2.imwrite('q4_09_highpass_cutoff.jpg', highpass_cutoff_show)

b = np.take(shifted_deer, 0, axis=2)
g = np.take(shifted_deer, 1, axis=2)
r = np.take(shifted_deer, 2, axis=2)

helper =  np.zeros(shifted_deer.shape)
helper = lowpass_cutoff[0:b.shape[0],0:b.shape[1]]


lowpassed_deer = np.zeros((b.shape[0],b.shape[1],3), dtype=np.complex)
lowpassed_deer[:,:,0] = b*helper[:,:]
lowpassed_deer[:,:,1] = g*helper[:,:]
lowpassed_deer[:,:,2] = r*helper[:,:]

b = np.take(shifted_lion, 0, axis=2)
g = np.take(shifted_lion, 1, axis=2)
r = np.take(shifted_lion, 2, axis=2)

helper =  np.zeros(shifted_lion.shape)
helper = highpass_cutoff[0:b.shape[0],0:b.shape[1]]

highpassed_lion = np.zeros((b.shape[0],b.shape[1],3),dtype=np.complex)
highpassed_lion[:,:,0] = b*helper[:,:]
highpassed_lion[:,:,1] = g*helper[:,:]
highpassed_lion[:,:,2] = r*helper[:,:]

lowpassed_deer_show = np.zeros(lowpassed_deer.shape, dtype=('uint8'))
highpassed_lion_show = np.zeros(highpassed_lion.shape, dtype=('uint8'))
lowpassed_deer_log_abs = np.log(np.abs(lowpassed_deer)+0.0001)
highpassed_lion_log_abs = np.log(np.abs(highpassed_lion)+0.0001)
cv2.normalize(highpassed_lion_log_abs,highpassed_lion_show,0,255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
cv2.normalize(lowpassed_deer_log_abs,lowpassed_deer_show,0,255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)

cv2.imwrite('q4_11_highpassed.jpg', highpassed_lion_show)
cv2.imwrite('Q4_12_lowpassed.jpg', lowpassed_deer_show)

deer_weight = 2
lion_weight = 1
total = deer_weight + lion_weight
first = (deer_weight/total)*lowpassed_deer
second =  (lion_weight/total)*highpassed_lion
hybrid_frequency = first+second

hybrid_frequency_log_abs = np.log(np.abs(hybrid_frequency)+0.0001)
hybrid_frequency_show = np.zeros(hybrid_frequency.shape, dtype=('uint8'))
cv2.normalize(hybrid_frequency_log_abs,hybrid_frequency_show,0,255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
cv2.imwrite('Q4_13_hybrid_frequency.jpg', hybrid_frequency_show)


hybrid_frequency_b = np.take(hybrid_frequency, 0, axis=2)
hybrid_frequency_g = np.take(hybrid_frequency, 1, axis=2)
hybrid_frequency_r = np.take(hybrid_frequency, 2, axis=2)

hybrid_ishift_b = np.fft.ifftshift(hybrid_frequency_b)
hybrid_ishift_g = np.fft.ifftshift(hybrid_frequency_g)
hybrid_ishift_r = np.fft.ifftshift(hybrid_frequency_r)

hybrid_image_b = np.fft.ifft2(hybrid_ishift_b)
hybrid_image_g = np.fft.ifft2(hybrid_ishift_g)
hybrid_image_r = np.fft.ifft2(hybrid_ishift_r)

hybrid_image_show_b = np.real(hybrid_image_b)
hybrid_image_show_g = np.real(hybrid_image_g)
hybrid_image_show_r = np.real(hybrid_image_r)

hybrid_image_show = cv2.merge([hybrid_image_show_b,hybrid_image_show_g, hybrid_image_show_r])
cv2.imwrite('Q4_14_hybrid_near.jpg', hybrid_image_show)
cv2.imwrite('Q4_15_hybrid_far.jpg', cv2.resize(hybrid_image_show,(int(hybrid_image_show.shape[1]/8),int(hybrid_image_show.shape[0]/8))))


