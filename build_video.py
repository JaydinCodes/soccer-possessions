'''
Get a sorted list of all the .jpg filenames in data/img1/.
Read the first image just to get its width/height (every video needs a fixed frame size).
Create a cv2.VideoWriter — this is the mirror image of VideoCapture: 
instead of reading frames from a video, it writes frames into one.
Loop through every image file in order, read it with cv2.imread, write it into the VideoWriter.

'''


import os
import cv2
image_folder = "data/img1"
output_path = "data/match.mp4"
fps = 25
filenames = sorted(os.listdir(image_folder))
first_frame = cv2.imread(os.path.join(image_folder, filenames[0]))
height, width, channels = first_frame.shape

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

for name in filenames:
    frame = cv2.imread(os.path.join(image_folder, name))
    writer.write(frame)


writer.release()
print("done")


