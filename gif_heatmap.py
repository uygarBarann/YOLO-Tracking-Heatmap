import os
from ultralytics import YOLO
import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt

def create_blank_heatmap(image_height, image_width):
    return np.zeros((image_height, image_width), dtype=np.float32)

def inc_heat_map(heatmaps, track_ids, boxes):
    for track_id, box in zip(track_ids, boxes):
        x, y, w, h = box
        heatmap = heatmaps.get(track_id, create_blank_heatmap(H, W)) #If there is no heatmap for the object corresponding track_id, create one; if there is, use that heatmap instead.
        #Increment the heat value of each pixel inside the bounding box area
        y1, y2 = max(int(y - h / 2), 0), min(int(y + h / 2), H)
        x1, x2 = max(int(x - w / 2), 0), min(int(x + w / 2), W)
        heatmap[y1:y2, x1:x2] += 1
        heatmaps[track_id] = heatmap

video_name = "./hardhat/hardhat"
video_path = os.path.join(f'{video_name}.mp4')

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second
ret, frame = cap.read()
H, W, _ = frame.shape #Shape of the frames
model_path = os.path.join('.', 'best.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

frame_count = 0
frames = []
heatmaps = {}

#For each frame of the video, extract the box coordinates and track_ids for corresponding frame. 
#Update the heatmaps according to those informations.
while ret:
    results = model.track(frame, persist=True,  classes=[0,7]) #only safety vest and hardats are detected.
    
    if frame_count % (fps // 2) == 0:  # Append two frames per second into a list
        im_array = results[0].plot()
        im_array_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        frames.append(im_array_rgb)

    # Get bounding boxes and track IDs from the results
    try:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Update heatmaps for each track_id
        inc_heat_map(heatmaps, track_ids, boxes)
    except AttributeError:  
        continue  
    finally:
        ret, frame = cap.read()
        frame_count += 1

cap.release()

# Save and display heatmaps
total_frames = (len(frames) / 2) * fps #total frame count
for track_id, heatmap in heatmaps.items():
    heatmap_image_path = f"{video_name}_heatmap_track_{track_id}.png"
    plt.imshow(heatmap, cmap='jet', interpolation='nearest', vmin=0, vmax=total_frames)
    plt.colorbar()
    plt.savefig(heatmap_image_path)
    plt.clf()  # Clear the figure for the next heatmap


imageio.mimsave(f"{video_name}.gif", frames, duration=0.5) #Save a gif from frames appended to the list
cv2.destroyAllWindows()
