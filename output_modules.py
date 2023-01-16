### Functions to display output of climbing data

## Setup
import numpy as np
import cv2
import progressbar as pb
import pose_track_module as pm
import matplotlib.pyplot as plt

class Progress:
    """
    Class to make progress bar easier

    Attributes
    ----------
    msg : str
        message for the progress bar to display
    maxVal : int
        number where loop is finished
    """
    def __init__(self, msg, maxVal):
        self.msg = msg
        self.maxVal = maxVal

        self.widgets = [msg, pb.Percentage(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
        self.timer = pb.ProgressBar(widgets=self.widgets, max_value=maxVal)

    def start(self):
        """Start timer"""
        self.timer.start()
        
    def update(self, idx):
        """Update timer"""
        self.timer.update(idx)
    
    def finish(self):
        """End Timer"""
        self.timer.finish()

    def newTimer(self, msg, maxVal):
        """
        Create new timer

        Parameters
        ----------
        msg : str
            message for the progress bar to display
        maxVal : int
            number where loop is finished
        """
        self.msg = msg
        self.maxVal = maxVal
        self.widgets = [msg, pb.Percentage(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
        self.timer = pb.ProgressBar(widgets=self.widgets, max_value=maxVal)

class OutputCreator():
    """
    Class used to help create gifs
    """
    def __init__(self):
        self.detector = pm.poseDetector() # initialize the pose tracker

    def concat_images(self, imga, imgb):
        """
        Combines two color image ndarrays side-by-side.

        Parameters
        ----------
        imga, imgb : numpy.ndarray
            two images to concatenate

        Output
        ------
        new_img : numpy.ndarray
            concatenated image
        """
        ha,wa = imga.shape[:2]
        hb,wb = imgb.shape[:2]
        max_height = np.max([ha, hb])
        total_width = wa+wb
        new_img = np.zeros(shape=(max_height, total_width, 3))
        new_img[:ha,:wa]=imga
        new_img[:hb,wa:wa+wb]=imgb
        return new_img

    def concat_n_images(image_list):
        """
        Combines N color images from a list of numpy.ndarrays
        """
        output = None
        for i, img in enumerate(image_list):
            if i==0:
                output = img
            else:
                output = OutputCreator.concat_images(output, img)
        return output

    def create_frames(self, videoCapture, draw=False):
        """
        Takes a cv2.VideoCapture object and creates frames for gif

        Parameters
        ----------
        videoCapture : cv2.VideoCapture 
            cv2 video capture object with video
        draw : boolean 
            draw pose onto image frames (default=false)

        Output
        ------
        numframes : int
            number of frames created
        imgframes : list of numpy.ndarray's
            frames of the original image (color video)
        plotframes : list of numpy.ndarray's
            frames of the 3d plot

        returns numframes, imgframes, plotframes
        """
        # create progress bar for plotting the figure
        framenum = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        widgets = ['Creating 3D plot: ', pb.Percentage(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
        timer_plt = pb.ProgressBar(widgets=widgets, max_value=framenum).start()

        ## Create frames of plotted figure
        plotframes = []
        imgframes = []
        idx=0
        while True:
            success, img = videoCapture.read()
            if not success:
                break
            plot = self.detector.find3DPose(img)
            if type(plot) != np.ndarray:
                plot.canvas.draw()
                gifframe = np.frombuffer(plot.canvas.tostring_rgb(), dtype=np.uint8)
                gifframe = gifframe.reshape(plot.canvas.get_width_height()[::-1] + (3,))
                idx += 1
                plotframes.append(gifframe)
                if draw:
                    imgframe = self.detector.findPose(img)
                    imgframes.append(cv2.cvtColor(imgframe, cv2.COLOR_BGR2RGB))
                else:
                    imgframes.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.close()
            timer_plt.update(idx)
        timer_plt.finish()

        return len(plotframes), imgframes, plotframes

    def get_detector(self):
        return self.detector