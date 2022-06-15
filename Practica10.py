import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
   
image = cv2.imread('img.jpg') 
imageReal = cv2.imread('img.jpg')
   
mask = np.zeros(image.shape[:2], np.uint8) 
   
backgroundModel = np.zeros((1, 65), np.float64) 
foregroundModel = np.zeros((1, 65), np.float64) 

rectangle = (10, 50, 500, 250) 
   
cv2.grabCut(image, mask, rectangle,backgroundModel, foregroundModel,3,cv2.GC_INIT_WITH_RECT) 
mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8') 
   

image = image * mask2[:, :, np.newaxis] 

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

esquinas = cv2.goodFeaturesToTrack(gray, 500, 0.01 ,10)
esquinas = np.int0(esquinas)

for corner in esquinas:
    x, y = corner.ravel()
    cv2.circle(image, (x,y), 3 , 255, -1)

cv2.imshow("Imagen", imageReal)
cv2.imshow("imagen Esquinas", image)

   
cv2.waitKey(0)
cv2.destroyAllWindows()