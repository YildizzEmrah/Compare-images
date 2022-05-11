import cv2 
from skimage.measure import compare_ssim
from skimage.measure import compare_mse as mse
from skimage import img_as_float
import math
import numpy as np

img1=cv2.imread('original-image.png')
img2=cv2.imread('processed-image.png')

def rmse(img1, img2):
        """Calculates the root mean square error (RSME) between two images"""
        return math.sqrt(mse(img_as_float(img1), img_as_float(img2))) *255

def psnr(img1,img2):
        """Calculates the peak signal-to-noise ratio (PSNR) between two images"""

        return cv2.PSNR(img1,img2)

def ssim(img1,img2):    
        """Calculates the structural similarity index measure (SSIM) between two images"""
        grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        (score, diff) = compare_ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        return score
    
def mse(img1,img2):
        """Calculates the mean square error (MSE) between two images"""
        return np.square(np.subtract(img1,img2)).mean()
rmse=rmse(img1,img2)
psnr=psnr(img1,img2)
ssim=ssim(img1,img2)
mse=mse(img1,img2)


print("Psnr : {:.3f} ; Ssim : {:.3f} ; Mse : {:.3f} ; Rmse : {:.3f} ".format(psnr,ssim,mse,rmse))
