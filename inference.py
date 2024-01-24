import pyrealsense2 as rs
import numpy as np
import cv2
import torch

#import YOLOX functions and attributes
# NOTE: find a way to remove redundant yolo files and keep yolox-nano only
from yolox import yolox_nano
from yolox.data.data_augment import ValTransform
#from yolox.data.datasets import COCO_CLASSES
from yolox.utils import postprocess, vis
from yolox.utils.visualize import plot_tracking

# import BYTETracker
from bytetrack import BYTETracker

# load yolox model and weights here
# COMMENT: these parameters are hardcoded we might wanna offload
#          offload them into a separate config file
test_size = (512, 512)
test_conf = 0.01
nmsthre = 0.65
num_classes = 80

model = yolox_nano.Exp().get_model()
ckpt = torch.load('weights/yolox_nano.pth', map_location="cpu")
model.load_state_dict(ckpt["model"])
model.cuda()
model.half()
model.eval()
print('COCO-pretrained yolox-nano loaded')

# yolox utils
preproc = ValTransform(legacy=False)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

width, height = 1280, 720
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

print('Realsense camera stream loaded')

# load BYTETracker
aspect_ratio_thresh = 0.6
min_box_area = 10
track_thresh = 0.5
track_buffer = 30
match_thresh = 0.8
fuse_score = False
frame_rate = 30

tracker = BYTETracker(track_thresh=track_thresh,
					  track_buffer=track_buffer,
					  match_thresh=match_thresh,
					  fuse_score=fuse_score,
					  frame_rate=frame_rate)

# Start streaming
pipeline.start(config)
with torch.no_grad():
	while True:
		# Wait for color frame
		frames = pipeline.wait_for_frames()
		color_frame = frames.get_color_frame()
		if not color_frame:
			continue

		# Convert images to numpy arrays then to torch tensors in GPU
		img = np.asanyarray(color_frame.get_data()) #BGR?? RGB?? Double check if channels are correct
		img_ori = img
		ratio = min(test_size[0] / height, test_size[1] / width)
		
		img, _ = preproc(img, None, test_size)
		img = torch.from_numpy(img).unsqueeze(0).float().half().cuda()
		
		# inference
		outputs = model(img)
		outputs = postprocess(outputs,
							  num_classes,
							  test_conf,
							  nmsthre,
							  class_agnostic=True)
		
		if outputs[0] is not None:
			output = outputs[0].cpu()
			output = output[output[:, 6] == 0] # filter out ONLY persons
			
			#BYTETracker
			online_targets = tracker.update(output, [height, width], test_size)
			online_tlwhs = []
			online_ids = []
			online_scores = []
			for t in online_targets:
				tlwh = t.tlwh
				tid = t.track_id
				vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
				if tlwh[2] * tlwh[3] > min_box_area and not vertical:
					online_tlwhs.append(tlwh)
					online_ids.append(tid)
					online_scores.append(t.score)
			# Plot BYTETracker output
			# print(online_tlwhs) use this to get scrops for tracked object
			vis_res = plot_tracking(img_ori,
									online_tlwhs,
									online_ids)
		else:
			vis_res = img_ori

		# Show images
		cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
		cv2.imshow('RealSense', vis_res)
		if cv2.waitKey(1) & 0xff == ord('q'):
			break

pipeline.stop()
