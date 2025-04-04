import cv2
#import opencv library ,used for videos and photo


#cAPTURING OUR FACIAL CLASSIFICATION 
face_cap=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# we added predefined data in cv2 file inside python file in my computer for facial detection i added frontal face file
#basically since our this data were stored in different folder we copied the path as well



#created video variable which will enable video 
# for runtime video we gave value of 0
video_cap=cv2.VideoCapture(0)




#we want to continue video until we press an key to stop for that

while True:    #now we have to read the run time images for that 
    ret, video_data=video_cap.read()
    #we made two varaiable aone we are using for reading the live image (.read()) method
    #for creatring the frame of video
    #ret tells whethr the frame was succesfully captured or not
    col=cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY)

# BAscially we changed our video color using cv2.cvtColor method into black and white color using cv2.COLOR_BGR2GRAY
#so that face recognition could capture the details of the face
    faces=face_cap.detectMultiScale(  #detecting the faces in the frame
        col, #gray color 
        scaleFactor=1.1,  #reduces images size by 10 % at each scale 
        minNeighbors=5, #the number of rectangel that should detect around a face 
        minSize=(40,40),  #thr minimum size of detected face
        flags= cv2.CASCADE_SCALE_IMAGE
        )
    for(x,y,w,h) in faces:       #(x,y),(x+w,y+h) the cordinates of the rectangle
        cv2.rectangle(video_data,(x,y),(x+w,y+h),(255,255,0),2) #2 is the thickness
    #if face is detected opencv returns the coordinates 
    #x,y -top left corner of the face
    #wh width and height

    cv2.imshow("camera",video_data)
    #using imshow  we are creating a frame and then we gave it a name and the the data
    if cv2.waitKey(10)== ord("t"):
        break
    #we used waitkey method to let run the image till given duration in milllisecond
    #ord used for the key press function giving ascii unicode 
    #basically when condition becomes true it break and stop the video
video_cap.release()





