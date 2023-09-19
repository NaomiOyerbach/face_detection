import cv2
from random import randrange


trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# img = cv2.imread('RONI0915.jpg')
webcam = cv2.imread('RONI0915.jpg')

# webcam = cv2.VideoCapture('vid.mp4')
# webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    print(face_coordinates)
    for ( x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y),(x+w,y+h), (0,255,0), 5)
        # print()
    cv2.imshow('Clever Programmer Face Detector', frame)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break
    
# webcam.release()
    
"""
# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# face_coordinates[0]
for ( x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y),(x+w,y+h), (randrange(256),randrange(256),randrange(256)), 2)

print(face_coordinates)


print("Code Completed")
"""

# import cv2
# from random import randrange


# trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# img = cv2.imread('RONI0915.jpg')

# # 
# grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Detect faces
# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# # face_coordinates[0]
# for ( x, y, w, h) in face_coordinates:
#     cv2.rectangle(img, (x, y),(x+w,y+h), (randrange(256),randrange(256),randrange(256)), 2)

# print(face_coordinates)

# # 
# cv2.imshow('Clever Programmer Face Detector', img)

# cv2.waitKey()

# print("Code Completed")


