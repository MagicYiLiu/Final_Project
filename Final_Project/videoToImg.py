import cv2

vc = cv2.VideoCapture(r'/Users/rick/Desktop/IT PROJECT/Final_Project/Final_Project/data/video/1.mp4')  # read video
n = 1  # count number

if vc.isOpened():  # Determine whether to open normally
    rval, frame = vc.read()
else:
    rval = False

timeF = 50  # Video frame count interval frequency

i = 0
while rval:  # Loop read video frame
    rval, frame = vc.read()
    if (n % timeF == 0):  # Store operation every timeF
        i += 1
        print(i)
        cv2.imwrite(r'/Users/rick/Desktop/IT PROJECT/Final_Project/Final_Project/data/toImage/{}.jpg'.format(i), frame)  # save as image
    n = n + 1
    cv2.waitKey(1)
vc.release()