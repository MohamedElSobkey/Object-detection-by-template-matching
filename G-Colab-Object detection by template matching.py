

import cv2 
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from skimage import io

#img = io.imread('./messi.jpg', 0)
img = cv2.imread('messi.jpg',1)
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.imread('messi_face.jpg', 0)
#template = io.imread('./messi_face.jpg', 0)

#-1 to return w before h 
w, h = template.shape[::-1]
res_img = cv2.matchTemplate(imgray,template,cv2.TM_CCORR_NORMED)
cv2_imshow( res_img)

print(res_img)

# to detect the location of the pixels
threshold = 0.86
loc = np.where(res_img>= threshold)
print (loc)


# zip(loc[1],loc[0]) 
  #  zip(*loc[::-1])  # -1 to inver 0 & 1

for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
     
     
cv2_imshow( img)
cv2.waitKey(0)
cv2.destroyAllWindows()