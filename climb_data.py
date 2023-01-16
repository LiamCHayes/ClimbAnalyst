### Generate data for a climbing report

## Setup
import cv2
import argparse
import math
import os
import matplotlib.pyplot as plt
import numpy as np
from output_modules import OutputCreator, Progress

# create argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=True, help='path to the video')
ap.add_argument('-n', '--name', required=True, help='name of output videos and directory')
ap.add_argument('-d', '--draw', required=False, default=True, help='draw pose over image')
ap.add_argument('-l', '--limbex', required=False, default=True, help='get limb extension graphs')
ap.add_argument('-e', '--velocity', required=False, default=True, help='get hand / arm velocity graphs')
ap.add_argument('-c', '--cog', required=False, default=True, help='get center of gravity video')
ap.add_argument('-s', '--smooth', required=False, default=3, help='amount of smoothing for the graphs')
args = vars(ap.parse_args())

# initialize objects
oc = OutputCreator()
pb = Progress(' ', 0)
detector = oc.get_detector()


## Get frames for image and plot
cap = cv2.VideoCapture(args['video']) 
numframes, imgframes, plotframes = oc.create_frames(cap)
cap.release()


## Get coordinates for each frame
pb.newTimer('Getting Coordinates: ', numframes)
pb.start()
lmlist = [None] * numframes
imglmlist = [None] * numframes
i=0
for im in imgframes:
    lmlist[i] = detector.findRelativePosition(im)
    imglmlist[i] = detector.findPosition(im)
    i+=1
    pb.update(i)
pb.finish()


## Arm and leg extension
def limbExtension():
    pb.newTimer('Calculating Limb Extension Values: ', numframes)
    pb.start()
    raextension = []
    laextension = []
    rlextension = []
    llextension = []
    raextensionMA = []
    laextensionMA = []
    rlextensionMA = []
    llextensionMA = []
    armExtensionFrames = [None] * numframes
    legExtensionFrames = [None] * numframes
    i=0
    for lm in lmlist:
        if lm != []:
            # Right arm
            rshoulder = lm[12]
            rhand = lm[16]
            raextension.append(math.sqrt((rshoulder[1] - rhand[1])**2 + (rshoulder[2] - rhand[2])**2 + (rshoulder[3] - rhand[3])**2))
            # Left arm
            lshoulder = lm[11]
            lhand = lm[15]
            laextension.append(math.sqrt((lshoulder[1] - lhand[1])**2 + (lshoulder[2] - lhand[2])**2 + (lshoulder[3] - lhand[3])**2))
            # Right leg
            rleg = lm[24]
            rfoot = lm[28]
            rlextension.append(math.sqrt((rleg[1] - rfoot[1])**2 + (rleg[2] - rfoot[2])**2 + (rleg[3] - rfoot[3])**2))
            # Left leg
            lleg = lm[23]
            lfoot = lm[27]
            llextension.append(math.sqrt((lleg[1] - lfoot[1])**2 + (lleg[2] - lfoot[2])**2 + (lleg[3] - lfoot[3])**2))
        else:
            if i != 0:
                raextension.append(raextension[i-1])
                laextension.append(laextension[i-1])
                rlextension.append(rlextension[i-1])
                llextension.append(llextension[i-1])
            else:
                raextension.append(0)
                laextension.append(0)
                rlextension.append(0)
                llextension.append(0)

        # smoothing function
        windowSize = int(args['smooth'])
        if i < windowSize:
            raextensionMA.append(raextension[i])
            laextensionMA.append(laextension[i])
            rlextensionMA.append(rlextension[i])
            llextensionMA.append(llextension[i])
        else:
            raextensionMA.append(sum(raextension[i-windowSize : i])/windowSize)
            laextensionMA.append(sum(laextension[i-windowSize : i])/windowSize)
            rlextensionMA.append(sum(rlextension[i-windowSize : i])/windowSize)
            llextensionMA.append(sum(llextension[i-windowSize : i])/windowSize)

        # leg extension plot
        fig = plt.figure()
        ax = plt.axes()
        plt.title("Leg Extension")
        plt.ylim(0, 1)
        x = np.linspace(0, i+1, i+1)
        ax.plot(x, rlextensionMA, label='Right Leg')
        ax.plot(x, llextensionMA, label='Left Leg')
        plt.legend()
        fig.canvas.draw()
        extframe = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        extframe = extframe.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        legExtensionFrames[i] = extframe
        plt.close()

        # arm extension plot
        fig = plt.figure()
        ax = plt.axes()
        plt.title("Arm Extension")
        plt.ylim(0, 1)
        x = np.linspace(0, i+1, i+1)
        ax.plot(x, raextensionMA, label='Right Arm')
        ax.plot(x, laextensionMA, label='Left Arm')
        plt.legend()
        fig.canvas.draw()
        extframe = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        extframe = extframe.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        armExtensionFrames[i] = extframe
        plt.close()
        
        i+=1
        pb.update(i)
    pb.finish()

    return armExtensionFrames, legExtensionFrames

        
## Hand and foot velocity chart
def HFvelo():
    pb.newTimer('Calculating Hand and Foot Velocity Values: ', numframes)
    pb.start()
    rhvelo = []
    lhvelo = []
    rfvelo = []
    lfvelo = []
    rhveloMA = []
    lhveloMA = []
    rfveloMA = []
    lfveloMA = []
    handVeloFrames = [None] * numframes
    footVeloFrames = [None] * numframes
    for i in range(numframes):
        if imglmlist[i] != [] and i != 0:
            rhvelo.append(math.sqrt((imglmlist[i][20][1]-imglmlist[i-1][20][1])**2 + (imglmlist[i][20][2]-imglmlist[i-1][20][2])**2)*30)
            lhvelo.append(math.sqrt((imglmlist[i][19][1]-imglmlist[i-1][19][1])**2 + (imglmlist[i][19][2]-imglmlist[i-1][19][2])**2)*30)
            rfvelo.append(math.sqrt((imglmlist[i][32][1]-imglmlist[i-1][32][1])**2 + (imglmlist[i][32][2]-imglmlist[i-1][32][2])**2)*30)
            lfvelo.append(math.sqrt((imglmlist[i][31][1]-imglmlist[i-1][31][1])**2 + (imglmlist[i][31][2]-imglmlist[i-1][31][2])**2)*30)
        else:
            if i != 0:
                rhvelo.append(rhvelo[i-1])
                lhvelo.append(lhvelo[i-1])
                rfvelo.append(rfvelo[i-1])
                lfvelo.append(lfvelo[i-1])
            else:
                rhvelo.append(0)
                lhvelo.append(0)
                rfvelo.append(0)
                lfvelo.append(0)

        # smoothing function
        windowSize = int(args['smooth'])
        if i < windowSize:
            rhveloMA.append(rhvelo[i])
            lhveloMA.append(lhvelo[i])
            rfveloMA.append(rfvelo[i])
            lfveloMA.append(lfvelo[i])
        else:
            rhveloMA.append(sum(rhvelo[i-windowSize : i])/windowSize)
            lhveloMA.append(sum(lhvelo[i-windowSize : i])/windowSize)
            rfveloMA.append(sum(rfvelo[i-windowSize : i])/windowSize)
            lfveloMA.append(sum(lfvelo[i-windowSize : i])/windowSize)

        # hand velocity plot
        fig = plt.figure()
        ax = plt.axes()
        plt.title("Arm Velocity")
        plt.ylim(0, 1000)
        x = np.linspace(0, i+1, i+1)
        ax.plot(x, rhveloMA, label='Right Hand')
        ax.plot(x, lhveloMA, label='Left Hand')
        plt.legend()
        fig.canvas.draw()
        extframe = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        extframe = extframe.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        handVeloFrames[i] = extframe
        plt.close()

        # foot velocity plot
        fig = plt.figure()
        ax = plt.axes()
        plt.title("Foot Velocity")
        plt.ylim(0, 1000)
        x = np.linspace(0, i+1, i+1)
        ax.plot(x, rfveloMA, label='Right Foot')
        ax.plot(x, lfveloMA, label='Left Foot')
        plt.legend()
        fig.canvas.draw()
        extframe = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        extframe = extframe.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        footVeloFrames[i] = extframe
        plt.close()

        pb.update(i)
    pb.finish()

    return handVeloFrames, footVeloFrames


## Center of gravity line
def COGline():
    pb.newTimer('Calculating Center Of Gravity Line: ', numframes)
    pb.start()
    cogFrames = imgframes.copy()
    for i in range(numframes):
        if imglmlist[i] != []:
            rhip = imglmlist[i][24]
            lhip = imglmlist[i][23]
            rshoulder = imglmlist[i][12]
            lshoulder = imglmlist[i][11]
            centerShoulder = (abs(rshoulder[1]-lshoulder[1])/2 + min(rshoulder[1], lshoulder[1]), abs(rshoulder[2]-lshoulder[2])/2 + min(rshoulder[2], lshoulder[2]))
            centerHips = (abs(rhip[1]-lhip[1])/2 + min(rhip[1], lhip[1]), abs(rhip[2]-lhip[2])/2 + min(rhip[2], lhip[2]))
            centerGravity = (int((centerShoulder[0]+centerHips[0])/2), int((centerShoulder[1]+centerHips[1])/2))
            # draw downward line and circle at COG
            cogFrames[i] = cv2.circle(cv2.line(cv2.line(cogFrames[i].copy(), centerGravity, (centerGravity[0], imgframes[0].shape[0]), (12, 199, 6), 10), centerGravity, (centerGravity[0], 0), (255, 255, 255), 3), centerGravity, radius=10, color=(199, 6, 6), thickness=-1)
    pb.finish()
    return cogFrames


## Get types of data requested
if args['limbex'] == True: armExtensionFrames, legExtensionFrames =  limbExtension()
if args['velocity'] == True: handVeloFrames, footVeloFrames = HFvelo()
if args['cog'] == True: cogframes = COGline()


## Output videos
fps = 30
baseVideoNum = 2
if args['draw'] == True: baseVideoNum += 1
if args['limbex'] == True: baseVideoNum += 2
if args['velocity'] == True: baseVideoNum += 2
if args['cog'] == True: baseVideoNum += 1
timerLen = numframes*baseVideoNum
pb.newTimer('Creating Videos: ', timerLen)
a=0
pb.start()
os.mkdir(f'./video_output/{args["name"]}')

# create plot video
h, w = plotframes[0].shape[:2]
out = cv2.VideoWriter(f'./video_output/{args["name"]}/plot.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
for i in range(numframes):
    out.write(cv2.cvtColor(plotframes[i], cv2.COLOR_RGB2BGR))
    a+=1
    pb.update(a)
out.release()

# create regular / drawn on video
h, w = imgframes[0].shape[:2]
out = cv2.VideoWriter(f'./video_output/{args["name"]}/raw_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
for i in range(numframes):
    out.write(cv2.cvtColor(imgframes[i], cv2.COLOR_RGB2BGR))
    a+=1
    pb.update(a)
out.release()
if args['draw'] == True:
    out = cv2.VideoWriter(f'./video_output/{args["name"]}/pose_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    for i in range(numframes):
        out.write(cv2.cvtColor((detector.findPose(imgframes[i], args['draw'])), cv2.COLOR_RGB2BGR))
        a+=1
        pb.update(a)
    out.release()

# create limb extension plots
if args['limbex'] == True:
    # create arm extension plot video
    h, w = armExtensionFrames[0].shape[:2]
    out = cv2.VideoWriter(f'./video_output/{args["name"]}/armextension.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    for i in range(numframes):
        out.write(armExtensionFrames[i])
        a+=1
        pb.update(a)
    out.release()

    # create leg extension plot video
    h, w = legExtensionFrames[0].shape[:2]
    out = cv2.VideoWriter(f'./video_output/{args["name"]}/legextension.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    for i in range(numframes):
        out.write(legExtensionFrames[i])
        a+=1
        pb.update(a)
    out.release()

# create hand / foot velocity plots
if args['velocity'] == True:
    # create hand velocity plot video
    h, w = handVeloFrames[0].shape[:2]
    out = cv2.VideoWriter(f'./video_output/{args["name"]}/handvelocity.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    for i in range(numframes):
        out.write(handVeloFrames[i])
        a+=1
        pb.update(a)
    out.release()

    # create foot velocity plot video
    h, w = footVeloFrames[0].shape[:2]
    out = cv2.VideoWriter(f'./video_output/{args["name"]}/footvelocity.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    for i in range(numframes):
        out.write(footVeloFrames[i])
        a+=1
        pb.update(a)
    out.release()

# create center of gravity video
if args['cog'] == True:
    h, w = cogframes[0].shape[:2]
    out = cv2.VideoWriter(f'./video_output/{args["name"]}/center_gravity.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    for i in range(numframes):
        out.write(cv2.cvtColor(cogframes[i], cv2.COLOR_RGB2BGR))
        a+=1
        pb.update(a)
    out.release()


## Finish creating video timer
pb.finish()