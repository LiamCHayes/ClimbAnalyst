### Pose tracking module

## Setup
import cv2
import mediapipe as mp
import time

class poseDetector():
    """
    Class used to extract poses from images

    Attributes
    ----------
    mode : bool
        treat input images as a batch of unrelated images as opposed to video stream (default=False)
    upBody : bool
        Just track upper body (default=False)
    smooth : bool
        Smooth landmarks to create less jitter (default=True)
    detectCon : bool
        detection confidence (default bool(0.5)) 
    trackCon : bool
        tracking confindence (default bool(0.5))
    """
    def __init__(self, mode=False, upBody=False, smooth=True, detectCon=bool(0.5), trackCon=bool(0.5)):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectCon = detectCon
        self.trackCon = trackCon

        ## Pose track module
        self.mpPose = mp.solutions.pose
        #print(self.mode, self.upBody, self.smooth, self.detectCon, self.trackCon)
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw = True):
        """
        Finds pose and draws it on the image
        
        Parameters
        ----------
        img : numpy.ndarray
            image to find pose
        draw : bool
            Draw pose on top of returned image (default=True)

        Output
        -------
        img : numpy.ndarray
            image with pose drawn on
        """
        if draw:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.pose.process(imgRGB)

            if self.results.pose_landmarks:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def find3DPose(self, img):
        """
        Finds pose and plots it on a 3d grid
        
        Parameters
        ----------
        img : numpy.ndarray
            image to find pose

        Output
        ------
        img : matplotlib plt plot
            plot of 3d pose
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_world_landmarks:
            img = self.mpDraw.plot_landmarks(self.results.pose_world_landmarks, self.mpPose.POSE_CONNECTIONS)
        
        return img

    def findPosition(self, img):
        """
        Finds list of position poses
        
        Parameters
        ----------
        img : numpy.ndarray
            image to find position coordinates relative to image

        Output
        ------
        lmList : list of coordinates for each body part where id is the id of each body part
            [id, x, y, z]
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        lmList = []
        if self.results.pose_landmarks:        
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy, depth = int(lm.x * w), int(lm.y * h), lm.z
                lmList.append([id, cx, cy, depth])
        
        return lmList

    def findRelativePosition(self, img):
        """
        Finds list of position poses

        Output
        ------
        lmList : list of coordinates for each body part where id is the id of each body part
            [id, x, y, z, visibility]
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        lmList = []
        if self.results.pose_world_landmarks:        
            for id, lm in enumerate(self.results.pose_world_landmarks.landmark):
                lmList.append([id, lm.x, lm.y, lm.z, lm.visibility])
        return lmList



def main():
    cap = cv2.VideoCapture('pose_videos/wm_settingvid2.mp4') # initialize video capture

    ptime = 0 # initialize variable to find fps

    detector = poseDetector()
    
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)

        # find fps
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime

        # show
        cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 3)
        cv2.imshow('image', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
