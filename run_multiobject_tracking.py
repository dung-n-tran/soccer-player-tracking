import sys
import cv2
from random import randint
from tqdm import tqdm
import time

def create_tracker_by_name(tracker_type="MIL"):
    if tracker_type == "BOOSTING":
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == "MIL":
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None

    return tracker

tracker = create_tracker_by_name()

filename = "102_3p30s_1080"
out_filepath = "out_" + filename + ".avi"
video = cv2.VideoCapture(filename+".mp4")

# Read first frame
success, frame = video.read()
if not success:
    print("Failed to read video")
video_size = (frame.shape[1], frame.shape[0])    
# Define ROI - Region of Interest
bboxes = []
colors = []
while True:
    k = cv2.waitKey(0) & 0xFF
    if (k == 113): # q is pressed
        break
    bbox = cv2.selectROI("MultiTracker", frame)
    bboxes.append(bbox)
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
cv2.destroyAllWindows()    
print("Selected bounding boxes {}".format(bboxes))

multi_tracker = cv2.MultiTracker_create()
tracking_boxes = []
for bbox in bboxes:
    multi_tracker.add(create_tracker_by_name("CSRT"), frame, bbox)
out_frames = []
start_tracking_time = time.time()
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    
    success, boxes = multi_tracker.update(frame)
    
    if success:
        for i, new_box in enumerate(boxes):
            p1 = (int(new_box[0]), int(new_box[1]))
            p2 = (int(new_box[0] + new_box[2]), int(new_box[1] + new_box[3]))
            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
        tracking_boxes.append(boxes)
    cv2.imshow("Multi Tracker", frame)
    out_frames.append(frame)
#     if cv2.waitKey(1) & 0xFF == 27: # Esc pressed
#         break

tracking_time = time.time() - start_tracking_time
print("Tracking time: {} [s]".format(tracking_time))

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
frame_per_second = 24
is_color = 1
writer = cv2.VideoWriter(out_filepath, fourcc, frame_per_second, video_size, is_color)
pbar = tqdm(range(len(out_frames)), unit="frame")

start_writing_time = time.time()
for i_frame in pbar:
    frame = out_frames[i_frame]
    writer.write(frame)
writing_time = time.time() - start_writing_time
print("Writing time: {} [s]".format(writing_time))
writer.release()
cv2.destroyAllWindows()     

import cPickle
with open(r"tracking_boxes.pickle", "wb") as output_file:
    cPickle.dump(tracking_boxes, output_file)