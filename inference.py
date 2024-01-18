import pyrealsense2 as rs
import numpy as np
import cv2
import torch

from yolox.exp import get_exp
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import fuse_model, get_model_info, postprocess, vis

# load model and weights here
exp = get_exp(None, 'yolox-nano')
model = exp.get_model()

ckpt = torch.load('weights/yolox_nano.pth', map_location="cpu")
model.load_state_dict(ckpt["model"])

model.cuda()
model.half()
model.eval()
print('COCO-pretrained yolox-nano loaded')

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print('Realsense camera stream loaded')

# utils
preproc = ValTransform(legacy=False)

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
		img = np.asanyarray(color_frame.get_data())
		img_ori = img
		ratio = min(exp.test_size[0] / img.shape[0], exp.test_size[1] / img.shape[1])
		
		img, _ = preproc(img, None, exp.test_size)
		img = torch.from_numpy(img).unsqueeze(0).float().half().cuda()
		
		#img_torch = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).half().cuda()
		
		# inference
		outputs = model(img)
		outputs = postprocess(outputs,
							  exp.num_classes,
							  exp.test_conf,
							  exp.nmsthre,
							  class_agnostic=True)
		output = outputs[0].cpu()
		output = output[output[:, 6] == 0] # filter out ONLY persons
		
		cls = output[:, 6]
		bboxes = output[:, 0:4]
		bboxes /= ratio
		scores = output[:, 4] * output[:, 5]
		vis_res = vis(img_ori, bboxes, scores, cls, 0.35, COCO_CLASSES)

		# Show images
		cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
		cv2.imshow('RealSense', vis_res)
		if cv2.waitKey(1) & 0xff == ord('q'):
			break

pipeline.stop()
