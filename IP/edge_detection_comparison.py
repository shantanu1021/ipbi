# https://www.javatpoint.com/dip-concept-of-edge-detection

import cv2
import numpy as np

''' Sobel edge detection method cannot produce smooth 
    and thin edge compared to canny method. But same like 
    other method, Sobel and Canny methods also very 
    sensitive to the noise pixels. Sometime all the noisy 
    image can not be filtered perfectly. Unremoved noisy pixels 
    will effect the result of edge detection.'''

def canny_edge_gen(img):
    name = 'Canny'
    # blur the image
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # Perform the canny operator
    canny = cv2.Canny(blurred, 30, 150)
    return canny, name

def sobel_edge_detection_operator(img):
    name = 'Sobel'
    Gx_kernel = np.array([[-1,0,1],
                          [-2,0,2],
                          [-1,0,1]])
    
    Gy_kernel = np.array([[1, 2, 1],
                          [0, 0, 0],
                          [-1,-2,-1]])

    img2 = cv2.filter2D(img, ddepth=-1, kernel=Gx_kernel)
    img2 = cv2.filter2D(img2, ddepth=-1, kernel=Gy_kernel)

    return img2, name

def robert_cross_operator(img, multiplier=1):
    name = 'Robert'
    Gx_kernel = multiplier*np.array([[1, 0],
                          [0,-1]])
    
    Gy_kernel = multiplier*np.array([[ 0 , 1],
                          [-1, 0]])
    
    print(Gx_kernel)

    img2 = cv2.filter2D(img, ddepth=-1, kernel=Gx_kernel)
    img2 = cv2.filter2D(img2, ddepth=-1, kernel=Gy_kernel)

    return img2,name

def laplacian_of_guassian(img,kernel=0):
    name = f"LoG - kernel_{kernel}"
    kernels =[np.array([[ 0,-1, 0],
                        [-1, 4,-1],
                        [ 0,-1, 0]]),
              np.array([[-1,-1,-1],
                        [-1, 8,-1],
                        [-1,-1,-1]]),
              np.array([[0, 1, 1,  2,  2,  2, 1, 1, 0],
                        [1, 2, 4,  5,  5,  5, 4, 2, 1],
                        [1, 4, 5,  3,  0,  3, 5, 4, 1],
                        [2, 5, 3,-12,-24,-12, 3, 5, 2],
                        [2, 5, 0,-24,-40,-24, 0, 5, 2],
                        [2, 5, 3,-12,-24,-12, 3, 5, 2],
                        [1, 4, 5,  3,  0,  3, 5, 4, 1],
                        [1, 2, 4,  5,  5,  5, 4, 2, 1],
                        [0, 1, 1,  2,  2,  2, 1, 1, 0]])]
    
    img2 = cv2.filter2D(img, ddepth=-1, kernel=kernels[kernel])
    return img2,name

img = cv2.imread(r'D:\\College\\IP\\img.jpg', 0)
cv2.imshow("Original", img)

names = []
images = []

image, name = canny_edge_gen(img)
images.append(image)
names.append(name)

image, name = sobel_edge_detection_operator(img)
images.append(image)
names.append(name)

image, name = robert_cross_operator(img, 2)
images.append(image)
names.append(name)

image, name = laplacian_of_guassian(img, 0)
images.append(image)
names.append(name)

image, name = laplacian_of_guassian(img, 1)
images.append(image)
names.append(name)

image, name = laplacian_of_guassian(img, 2)
images.append(image)
names.append(name)

for i in range(len(names)):
    cv2.imshow(names[i], images[i])
    if i<len(names)//2:
        cv2.moveWindow(names[i],i*img.shape[1],0)
    else:
        cv2.moveWindow(names[i],(i-len(names)//2)*img.shape[1],img.shape[0])

cv2.imshow("Original", img)
cv2.moveWindow("Original", 1350,200)

cv2.waitKey(0)