import cv2
import matplotlib.pyplot as plt
import numpy as np


class ImageSampling:
    def __init__(self,path):
        self.path = path
        self.img = cv2.imread(self.path)

    def showImg(self):
        cv2.imshow(str(self.img.shape)+" (Original)", self.img)

    def subsample(self,factor):
        img2 = np.zeros((self.img.shape[0]//factor, self.img.shape[1]//factor, self.img.shape[2]), dtype=np.uint8)
    
        for i in range(img2.shape[0]):
            for j in range(img2.shape[1]):
                for k in range(img2.shape[2]):
                   
                    img2[i,j,k] = self.img[i*factor,j*factor, k]

        cv2.imshow(str(img2.shape), img2)
        return img2

    def resample(self,img,factor):
        img2 = np.zeros((img.shape[0]*factor, img.shape[1]*factor,img.shape[2]), dtype=np.uint8)
        for i in range(img2.shape[0]):
            for j in range(img2.shape[1]):
                for k in range(img2.shape[2]):
                    
                    img2[i,j,k] = img[i//factor,j//factor, k]
        cv2.imshow(str(img.shape[:2])+"==>"+str(img2.shape[:2]), img2)
        cv2.moveWindow(str(img.shape[:2])+"==>"+str(img2.shape[:2]), img2.shape[0],0)
        return img2
    

class Histogram:
    def __init__(self,path):
        self.path = path

    def createHist(self):

        img = cv2.imread(self.path, 0)
        self.hist = cv2.calcHist([img],[0],None,[256],[0,256])

        equ = cv2.equalizeHist(img)
        self.equ_hist = cv2.calcHist([equ],[0],None,[256],[0,256])

        result = np.hstack((img, equ))

        cv2.imshow('image', result)
        self.plotHist()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plotHist(self):

        plt.plot(self.hist)
        plt.plot(self.equ_hist)
        plt.xlabel('No. of pixels')
        plt.ylabel('Value of pixels')
        plt.legend(["Normal", "Equilized"], loc ="upper right")
        plt.show()

imageSampling = ImageSampling(r'D:\\College\\IP\\img.jpg')

imageSampling.showImg()

img512 = imageSampling.subsample(2)
img256 = imageSampling.subsample(4)
img128 = imageSampling.subsample(8)
img64 = imageSampling.subsample(16)
img32 = imageSampling.subsample(32)

cv2.waitKey(0)

img1024 = imageSampling.resample(img32, 32)

cv2.waitKey(0)

histogram = Histogram(r'D:\\College\\IP\\img.jpg')
histogram.createHist()