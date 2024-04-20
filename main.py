from tkinter import *
from tkinter.messagebox import *
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

class ai:
    def main(self):
        root = Tk()
        root.geometry('550x600')
        root.title('Hand-Sign-Language-Detection-Application')
        
        # Title in the center
        Label(root, text="AMERICAN SIGN LANGAGUE  ", font=('arial',18 , 'bold')).grid(row=0, column=0,columnspan=500,pady=25)

        def destroy1():
            root.destroy()
            self.desc()
        def destroy2():
            root.destroy()
            self.helpp()
        def destroy3():
            root.destroy()
            self.start()

        #Label(root,text=" ").grid(row=2,column=0)
        #Label(root,text=" ").grid(row=2,column=1)

        handImage = PhotoImage(file='home_image.png')
        Label(root, image=handImage).grid(row=2, column=0,columnspan=10,padx=0)
        Label(root, text=" ").grid(row=3,column=0,pady=23)
        fr=Frame(root)
        fr.grid(row=4,column=0,columnspan=7,padx=95)
        Button(fr, text='Description', command=destroy1, font=('arial' , 11), bg='aquamarine').grid(row=4, column=2)
        Label(fr, text=" ").grid(row=4,column=3,padx=100)
        Button(fr, text='Guide', command=destroy2,font=('arial' , 11),width=6, bg='aquamarine').grid(row=4, column=5)
        Button(fr, text="Let's try!!", command=destroy3, font=('arial' , 11),height=2,width=12, bg='orange').grid(row=5, column=3,pady=40)
        
        root.mainloop()

    def desc(self):
        root = Tk()
        root.title('Description')
        root.geometry('390x350')
        
        Label(root, text="Description",font=('arial',11,'bold')).grid(row=0, column=0,padx=10, pady=10)
        Label(root, text="The project is a real-time Sign Langague detection application",font=('arial',9)).grid(row=1,column=0,padx=10)
        Label(root, text="developed using Python. The goal of this project was to build a ",font=('arial',9)).grid(row=2, column=0,padx=10 )
        Label(root, text="neural network able to classify which letter of the American Sign ",font=('arial',9)).grid(row=3, column=0,padx=10 )
        Label(root, text="Language alphabet is being signed, given an image of a signing ",font=('arial',9)).grid(row=5, column=0,padx=10)
        Label(root, text="hand. This project is a first step towards building a possible",font=('arial',9)).grid(row=6, column=0,padx=10 )
        Label(root, text="sign language translator, which can take communications in sign",font=('arial',9)).grid(row=8, column=0,padx=10 )
        Label(root, text="language and translate them into written language. Such a",font=('arial',9)).grid(row=9, column=0,padx=10 )
        Label(root, text="translator would greatly lower the barrier for many deaf and",font=('arial',9)).grid(row=10, column=0,padx=10 )
        Label(root, text="mute individuals to be able to better communicate with others",font=('arial',9)).grid(row=12, column=0,padx=10 )
        Label(root, text="      in day to day interactions.     ",font=('arial',9)).grid(row=13, column=0,padx=10 )
       

        def back():
            root.destroy()
            self.main()
        
        Button(root, text='Back', command=back,font=('arial',9,'bold')).grid(row=18,column=0,pady=15)
        root.mainloop()

    def helpp(self):
        root = Tk()
        root.title('Help')
        root.geometry('500x500')
        Label(root, text="Here are the ASL hand gestures which can be detected:",font=('arial',10,'bold')).grid(row=0,column=0,pady=5,padx=20)

        #Label(root, text="* Okay ,V (Peace), Thumbs up, Thumbs down, Call me,",font=('arial',9)).grid(row=1, column=0,padx=20)
        #Label(root, text="Stop, Rock, Live Long, Fist, Smile.",font=('arial',9)).grid(row=2, column=0,padx=20)
        Label(root, text="* To quit the webcam screen enter 'q'.",font=('arial',9)).grid(row=3, column=0,padx=20)
        # Image on the right-hand side in the middle
        helpImage = PhotoImage(file='guide.png')
        Label(root, image=helpImage).grid(row=4, column=0,pady=20,padx=20)
        
        def back():
            root.destroy()
            self.main()
        
        Button(root, text='Back', command=back,font=('arial',10)).grid(row=5,column=0,padx=20)
        root.mainloop()

    def start(self):
        root = Tk()
        root.title('Start Hand Gesture Detection')
        cap = cv2.VideoCapture(0)
        detector = HandDetector(maxHands=1)
        classifier = Classifier("Model/keras_model.h5","Model/labels.txt")

        offset = 20
        imgSize = 300

        labels=["A","B","C"]

        while True:
            success, img = cap.read()
            imgOutput = img.copy()
            hands, img = detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                imgWhite = np.ones((imgSize, imgSize, 3),np.uint8)*255
                imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

                imgCropShape = imgCrop.shape

                aspectRatio = h/w

                if aspectRatio>1:
                    k = imgSize/h
                    wCal = math.ceil(k*w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize-wCal)/2)
                    imgWhite[:, wGap:wCal+wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw = False)
                    ##print(prediction, index)

                else:
                    k = imgSize/w
                    hCal = math.ceil(k*h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize-hCal)/2)
                    imgWhite[hGap:hCal+hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw = False)
                    ##print(prediction, index)           

                cv2.rectangle(imgOutput, (x-offset,y-offset-50), (x-offset+90,y-offset), (51,68,255), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x,y-26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255,255,255), 2)
                cv2.rectangle(imgOutput, (x-offset,y-offset), (x+w+offset,y+h+offset), (51,68,255), 4)
                
                cv2.imshow("Hand Keypoints", imgCrop)
                ##cv2.imshow("ImageWhite", imgWhite)
         
            cv2.imshow("Hand-Sign-Detection-ASL",imgOutput)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()




        
        Label(root, text="Thank You!!").pack(pady=20)
        def back():
            root.destroy()
            self.main()
        
        Button(root, text='Back', command=back).pack(pady=10)
        root.mainloop()

a = ai()
a.main()
