import cv2
import numpy as np
import os 
import scipy.spatial as scp
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def read_points(file_points):
    lines = file_points.readlines() 
    n = int(lines[0])
    points_x = np.zeros((1,n), dtype = 'float32')
    points_y = np.zeros((1,n), dtype = 'float32')
    for i in range(0,n): 
        points_x[0][i] = float(lines[i+1].split(" ")[0])
        points_y[0][i] = float(lines[i+1].split(" ")[1])
    return  np.vstack([points_y, points_x]).T


lion = cv2.imread( os.getcwd()+'\\'+'lion.jpg',cv2.IMREAD_UNCHANGED)
deer = cv2.imread( os.getcwd()+'\\'+'deer.jpg',cv2.IMREAD_UNCHANGED)


file_points = open( 'lion-points.txt' ,mode = 'r')
lion_points = read_points(file_points)

file_points = open( 'deer-points.txt' ,mode = 'r')
deer_points = read_points(file_points)

helper_1 = deer[:,:,0].T
helper_2 = deer[:,:,1].T
helper_3 = deer[:,:,2].T
deer = np.zeros((deer.shape[1],deer.shape[0],3))
deer[:,:,0] = helper_1[:,:]
deer[:,:,1] = helper_2[:,:]
deer[:,:,2] = helper_3[:,:]

helper_1 = lion[:,:,0].T
helper_2 = lion[:,:,1].T
helper_3 = lion[:,:,2].T
lion = np.zeros((lion.shape[1],lion.shape[0],3))
lion[:,:,0] = helper_1[:,:]
lion[:,:,1] = helper_2[:,:]
lion[:,:,2] = helper_3[:,:]


triangles = scp.Delaunay(deer_points)
tri_deer = triangles.simplices

satr = np.arange(0, lion.shape[0])
satr = np.tile(satr, (lion.shape[1], 1)).T
sotun = np.arange(0, lion.shape[1])
sotun = np.tile(sotun, (lion.shape[0],1))

# positions = np.vstack([satr.ravel(), sotun.ravel()]).T
# correspondence  =  np.zeros((deer.shape[1],deer.shape[0]), dtype = 'uint8')
# positions = np.reshape(positions, (deer.shape[1], deer.shape[0], 2))
# correspondence  = scp.Delaunay.find_simplex(triangles, positions)

# itr_num = 80
# itr_num = 130
itr_num = 10
for t in range(0,itr_num+1):
    a = t/itr_num

    current_points = np.zeros(deer_points.shape, dtype = 'float32')
    current_points = a*(lion_points) + (1-a)*deer_points
    current_frame = np.zeros((deer.shape[1],deer.shape[0],3), dtype = 'float32')

    
    for i in range(0, tri_deer.shape[0]):
        index = tri_deer[i]

        src_deer = np.array([deer_points[index[0]],deer_points[index[1]] ,deer_points[index[2]]], dtype= 'float32')
        dst = np.array([current_points[index[0]],current_points[index[1]] ,current_points[index[2]]], dtype= 'float32')
        src_lion = np.array([lion_points[index[0]],lion_points[index[1]] ,lion_points[index[2]]], dtype= 'float32')
       
        mask =  np.zeros((deer.shape[1],deer.shape[0],3), dtype = 'uint8')

        pts = np.zeros(dst.shape)
        pts[:,0] = dst[:,1]
        pts[:,1] = dst[:,0]

        mask = cv2.fillPoly(mask,  np.int32([pts]), color=(1, 1, 1))
        mak = np.equal(current_frame, 0)
        mask *= mak

        pts = np.zeros(dst.shape).astype('float32')
        pts[:,0] = dst[:,1]
        pts[:,1] = dst[:,0]
        dst = pts
        matrix_deer = cv2.getAffineTransform(src_deer,dst)

        deer_warped = cv2.warpAffine(deer, matrix_deer, (deer.shape[0], deer.shape[1]))

        deer_warped = deer_warped*mask
  
        matrix_lion = cv2.getAffineTransform(src_lion,dst)

        lion_warped = cv2.warpAffine(lion, matrix_lion, (lion.shape[0], lion.shape[1]))
        lion_warped = lion_warped*mask
 
        first = a*lion_warped
        second = (1-a)*deer_warped

        current_frame += mask*(first + second).astype('uint8')

    cv2.imwrite(str(t).zfill(3)+'.jpg', current_frame)
    cv2.imwrite(str(itr_num*2 - t + 8).zfill(3)+'.jpg', current_frame)
    




   
           
        

