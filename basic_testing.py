import cv2
import face_recognition
 
imgElon = face_recognition.load_image_file('ImagesBasic/p1_real.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/p1_test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
 
faceLoc = face_recognition.face_locations(p1_real)[0]
encodereal = face_recognition.face_encodings(p1_real)[0]
cv2.rectangle(p1_real,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
 
faceLocTest = face_recognition.face_locations(p1_test)[0]
encodeTest = face_recognition.face_encodings(p1_test)[0]
cv2.rectangle(p1_test,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
 
result = face_recognition.compare_faces([encodereal],encodeTest)
faceDis = face_recognition.face_distance([encodereal],encodeTest)

cv2.putText(imgTest,f'{result} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
 
cv2.imshow('Real',p1_real)
cv2.imshow('Test',p1_test)
cv2.waitKey(0)
