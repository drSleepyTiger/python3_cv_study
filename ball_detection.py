import numpy as np
import cv2

cap = cv2.VideoCapture(0)
print('OpenCV',cv2.__version__)

while(1):

    ret, frame = cap.read()

    frame_blur = cv2.medianBlur(frame, 5)
    frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)


    circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=50, maxRadius=100)

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('frame', frame)
    cv2.imshow('frame_blur', frame_blur)
    cv2.imshow('frame_gray', frame_gray)

    if cv2.waitKey(0) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()