### Create gif for 3d plot

## Setup
import cv2
import argparse
import imageio
import matplotlib.pyplot as plt
import numpy as np
import progressbar as pb
from output_modules import OutputCreator

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=True, help='path to the video')
ap.add_argument('-d', '--draw', required=False, help='draw pose over image')
ap.add_argument('-n', '--name', required=True, help='name of output file')
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args['video']) # initialize video capture
gc = OutputCreator()

## Create frames for gif
numframes, imgframes, gifframes = gc.create_frames(cap, draw=args['draw'])

## Combine regular image with 3d plot
# create progress bar for stitching the figure and video
widgets = ['Stitching Images: ', pb.Percentage(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
timer_stitch = pb.ProgressBar(widgets=widgets, max_value=numframes).start()

# stitch images together
combined = []
for i in range(numframes):
    combinedimg = gc.concat_images(imgframes[i], gifframes[i]).astype("uint8")
    combined.append(combinedimg)
    timer_stitch.update(i)
timer_stitch.finish()

## Save gif
print('Creating GIF...')
imageio.mimsave(f"./3dplot_output/{args['name']}.gif", combined, fps=30)
