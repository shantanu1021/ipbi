import cv2
import numpy as np

class ContrastStretching:
    def __init__(self,path):
        self.path = path
        self.img = cv2.imread(r'D:\\College\\IP\\img.jpg', 0)

    def showImg(self):
        cv2.imshow("Original", self.img)

    '''
    MinI = Min pixel value of input image
    MaxI = Max pixel value of input image
    MinO = Min pixel value of output image
    MaxO = Min pixel value of output image
    '''
    def contrast_stretching_v2(self, MinO,MaxO):
        MinI = self.img.min()
        MaxI = self.img.max()
        print(MinI, MaxI)
        img_new = np.zeros((self.img.shape[0], self.img.shape[1]))

        for i in range(img_new.shape[0]):
            for j in range(img_new.shape[1]):
                img_new[i,j] = (self.img[i,j] - MinI) * (((MaxO-MinO)/(MaxI-MinI)) + MinO)
        return img_new

    '''
    Given pixel value r1, change it to s1 in new image
    Given pixel value r2, change it to s2 in new image
    '''
    def contrast_stretching(self, s1, r1, s2, r2):
        mini = self.img.min()
        maxi = self.img.max()
        print(mini, maxi)

        alpha = float(s1/r1)
        beta = float((s1-s2)/(r1-r2))
        gamma = float((maxi-s2)/(maxi-r2))

        img_new = np.zeros((self.img.shape[0], self.img.shape[1]))

        for i in range(img_new.shape[0]):
            for j in range(img_new.shape[1]):
                if self.img[i,j] < r1:
                    img_new[i,j] = alpha*self.img[i,j] 
                elif self.img[i,j] >= r1 and self.img[i,j] < r2:
                    img_new[i,j] = beta*(self.img[i,j] - r1) + s1
                else:
                    img_new[i,j] = gamma*(self.img[i,j] - r2) + s2
        return img_new

class IntensitySlicing:

    def __init__(self,path):
        self.path = path
        self.img = cv2.imread(r'D:\\College\\IP\\img.jpg', 0)

    def showImg(self):
        cv2.imshow("Original", self.img)

    def intensity_level_slicing(self, lower_val, upper_val, intensity):
        img_new = self.img
        for i in range(img_new.shape[0]):
            for j in range(img_new.shape[1]):
                if self.img[i,j] >= lower_val and self.img[i,j] <= upper_val:
                    img_new[i,j] = intensity
        return img_new


contrastStretching = ContrastStretching(r'D:\\College\\IP\\img.jpg')
intensitySlicing = IntensitySlicing(r'D:\\College\\IP\\img.jpg')

contrastStretching.showImg()


img_cs = contrastStretching.contrast_stretching(0, 113, 153, 230)
img_ils = intensitySlicing.intensity_level_slicing(100,180, intensity=12)
cv2.imshow("constrast stretching", img_cs)
cv2.imshow("intensity level slicing", img_ils)
cv2.waitKey(0)

