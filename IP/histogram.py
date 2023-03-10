import cv2
import matplotlib.pyplot as plt
import numpy as np


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
        
histogram = Histogram(r'D:\\College\\IP\\img.jpg')
histogram.createHist()
