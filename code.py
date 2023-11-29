
import cv2 
import numpy as np
import matplotlib.pyplot as plt

xcord=[]
def click_event(event, x, y, flags, params): 
   
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        print(x, ' ', y)
        xcord.append(int(x)) 

cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
im = cv2.imread("image1(2).jpg")                    # Read image
imS = cv2.resize(im,(1046,1080))                # Resize image
cv2.imshow("Original Image", imS)                       # Show image
cv2.waitKey(0)    




gray=cv2.cvtColor(imS,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (9, 9), 2)
lower_th = 100
upper_th = 150
th_mask = cv2.inRange(blurred, lower_th, upper_th)
thresh = cv2.bitwise_and(imS, imS, mask=th_mask)

r1= cv2.selectROI("Crop Femoral Head and Acetabelum for 'A' Measurement", thresh)
cropA=thresh[int(r1[1]):int(r1[1]+r1[3]),int(r1[0]):int(r1[0]+r1[2])] 
# cv2.imshow("Cropped",cropA) 
# cv2.waitKey(0)




corner=cropA.copy()
img=cv2.cvtColor(imS,cv2.COLOR_BGR2GRAY)
crop=cv2.cvtColor(cropA,cv2.COLOR_BGR2GRAY)

#Harris Croner Detection

gray = np.float32(crop.copy())
dst = cv2.cornerHarris(gray,3,3,0.04)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
corner[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow('Click on the detected corner of acetabelum and left most corner of femoral head',corner)
cv2.setMouseCallback('Click on the detected corner of acetabelum and left most corner of femoral head', click_event)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
A=abs(xcord[0]-xcord[1])
print("A=",A)

r= cv2.selectROI('Crop Femoral Head for "B"', thresh)
cropB=thresh[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])] 
# cv2.imshow("Femoral Head for 'B'",cropB) 
# cv2.waitKey(0)

cropB=cv2.cvtColor(cropB,cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(cropB.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cont=cv2.cvtColor(cropB,cv2.COLOR_GRAY2RGB).copy()

cont=cv2.drawContours(cont, contours,-1,color=(255, 0, 0),thickness= 2)
c=max(contours,key=cv2.contourArea)
x,y,w,h=cv2.boundingRect(c)
B=w
print("B=",B)
box=cont.copy()
cv2.rectangle(box,(x,y),(x+w,y+h),(0,255,0),2)

plt.subplot(221),plt.imshow(img,cmap='gray'),plt.colorbar()
plt.title('Original Image')

plt.subplot(222),plt.imshow(box)
plt.title('Box for calculation of B')

plt.subplot(223),plt.imshow(corner)
plt.title('Detected Corners')

plt.subplot(224),plt.imshow(cont)
plt.title('Contours for Femoral Head Detection'), plt.xticks([]), plt.yticks([])

plt.show()

cv2.destroyAllWindows() 
print("Calculated Migration Index is",A/B*100)

             


