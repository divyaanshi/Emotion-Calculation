import cv2,os
import cv2
from tkinter import *
import numpy as np
from PIL import Image
from ctypes import *
import sqlite3
import boto3
import datetime

recognizer=cv2.face.LBPHFaceRecognizer_create()
detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def MessageBox(title, text, style):  #Message box defination
    sty = int(style) + 4096
    return windll.user32.MessageBoxA(0, text, title, sty) #MB_SYSTEMMODAL==4096\

conn = sqlite3.connect('DY_DB1.db') #Opening Employee Database
conn1 = sqlite3.connect('EC_DY_DB.db') #Opening TS_DATABASE Database
                                            #Creating table TS_DATA in TS_DATABASE
conn1.execute('''CREATE TABLE IF NOT EXISTS EC_DY
         (TS_ID    INT     NOT NULL,
          NAME            STRING  NOT NULL,
          DATE            DATE    NOT NULL,
          TIME            TIME    NOT NULL,
          EMOTION         STRING  NOT NULL);''')


def getImagesWithID(path):


# get the path of all the files in the folder
   imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
# create empth face list
   faceSamples = []
# create empty ID list
   Ids = []
# now looping through all the image paths and loading the Ids and the images
   for imagePath in imagePaths:

    # Updates in Code
    # ignore if the file does not have jpg extension :
      if (os.path.split(imagePath)[-1].split(".")[-1] != 'jpg'):
          continue

        # loading the image and converting it to gray scale
      pilImage = Image.open(imagePath).convert('L')
    # Now we are converting the PIL image into numpy array
      imageNp = np.array(pilImage, 'uint8')
    # getting the Id from the image
      Id = int(os.path.split(imagePath)[-1].split(".")[1])
    # extract the face from the training image sample
      faces = detector.detectMultiScale(imageNp)
    # If a face is there then append that in the list as well as Id of it
      for (x, y, w, h) in faces:
        faceSamples.append(imageNp[y:y + h, x:x + w])
        Ids.append(Id)
   return faceSamples, Ids

faces,Ids = getImagesWithID('DataStorage')
recognizer.train(faces, np.array(Ids))
recognizer.save('DY_trainer.yml')
cv2.destroyAllWindows()

recognizer.read('DY_trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
cursor = conn.execute("SELECT ID, NAME, GENDER, DESIGNATION, SHIFT, CONTACT_NO from DY_DETAILS")

for row in cursor:
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            # cv2.rectangle(im,(x,y),(x+w,y+h),(127,255,0),2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            cursor = conn.execute("SELECT ID, NAME, GENDER, DESIGNATION, SHIFT, CONTACT_NO from DY_DETAILS")
            for row in cursor:
                if (conf < 40):
                    if (Id == row[0]):
                        now = datetime.datetime.now()

                        try:
                            ret, img = cam.read()
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the colored image into gray
                            faces = detector.detectMultiScale(gray, 1.3, 5)
                            sampleNum = 0
                            for (x, y, w, h) in faces:
                               cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                            # incrementing sample number
                               sampleNum = sampleNum + 1

                               DG = str(Id)
                            # saving the captured face in the dataset folder
                               cv2.imwrite("DataStorage/User." + DG + '.' + str(sampleNum) + ".jpg",
                                        gray[y:y + h, x:x + w])
                               print("Photo Saved")
                        except:
                               print("Exception Occured")

                        DY = str(Id)
                        imageFile = 'DataStorage/User.' +DY+ '.1.jpg'
                        client = boto3.client('rekognition')
                        with open(imageFile, 'rb') as image:
                            response = client.detect_faces(Image={'Bytes': image.read()}, Attributes=['ALL'])
                            b = 0
                            for emotion in response['FaceDetails']:

                                for sentiment in emotion['Emotions']:
                                    a = sentiment['Confidence']

                                    if b < a:
                                        b = a
                                        c = sentiment['Type']

                                emotion=c
                        """ conn = sqlite3.connect('Timestamp_Data.db')
                        print "Opened database successfully";"""
                        now = datetime.datetime.now()
                        conn1.execute(
                            "INSERT INTO EC_DY (TS_ID,NAME,DATE, TIME, EMOTION )Values (?,?,?,?,?)",
                            (str(Id), str(row[1]), now.strftime("%Y-%m-%d "), now.strftime("%H:%M"), emotion))

                        print("record added successfully")
                        conn1.commit()

                        # print "Records created successfully";
                        cv2.waitKey(100) == ord('p')

                        conn.close
                        conn1.close
                        # MessageBox('FACE RECOGNITION', 'id:'+str(row[0])+'\nName:'+str(row[1])+'\nGender:'+str(row[2])+'\nDESIGNATION:'+str(row[3])+'\nSHIFT:'+str(row[4])+'\nContact_No:'+str(row[5])+'\n'+now.strftime("%d-%m-%Y %H:%M")  , 64)
                else:
                    Id = "Unknown"
                    cv2.putText(im, str(Id), (x, h), font, 1, (255, 255, 255), 2)

        cv2.imshow('im', im)
        if cv2.waitKey(100) == ord('q'):
            break;
cam.release()
cv2.destroyAllWindows()



