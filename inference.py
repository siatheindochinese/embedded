import pyrealsense2 as rs
import numpy as np
from collections import OrderedDict
from queue import Queue
import cv2
import torch

# import YOLOX functions and attributes
# NOTE: find a way to remove redundant files
from yolox import yolox_nano
from yolox.data.data_augment import ValTransform
#from yolox.data.datasets import COCO_CLASSES
from yolox.utils import postprocess, vis
from yolox.utils.visualize import plot_tracking

# import BYTETracker
from bytetrack import BYTETracker

# import ViFi-CLIP
from vificlip.utils.config import get_config
from vificlip.trainers import vificlip
from vificlip.datasets.pipeline import Compose

################################################################
# load yolox model and weights here                            #
# COMMENT: these parameters are hardcoded, might wanna offload #
#          offload them into a separate config file            #
################################################################
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

# yolox utils
preproc = ValTransform(legacy=False)

###############################################
# Configure realsense depth and color streams #
###############################################
pipeline = rs.pipeline()
config = rs.config()

width, height = 1280, 720
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

####################
# load BYTETracker #
####################
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

################################################################
# load ViFi-CLIP, config and weights here                      #
# COMMENT: these parameters are hardcoded, might wanna offload #
#          offload them into a separate config file            #
################################################################
class_names = ['punch','kick','walk']

class parse_option():
    def __init__(self):
        self.config = 'vificlip/configs/zero_shot/train/k400/16_16_vifi_clip.yaml'
        self.output =  ''
        self.resume = 'weights/vifi_clip_10_epochs_k400_full_finetuned.pth'
        # No need to change below args.
        self.only_test = True
        self.opts = None
        self.batch_size = None
        self.pretrained = None
        self.accumulation_steps = None
        self.local_rank = 0
vargs = parse_option()
vconfig = get_config(vargs)

vmodel = vificlip.returnCLIP(vconfig,
                            class_names=class_names,)
vmodel.cuda()
vmodel.eval()
vmodel.half()

vckpt = torch.load(vconfig.MODEL.RESUME, map_location='cpu')
load_state_dict = vckpt['model']
# now remove the unwanted keys:
if "module.prompt_learner.token_prefix" in load_state_dict:
	del load_state_dict["module.prompt_learner.token_prefix"]

if "module.prompt_learner.token_suffix" in load_state_dict:
	del load_state_dict["module.prompt_learner.token_suffix"]

if "module.prompt_learner.complete_text_embeddings" in load_state_dict:
	del load_state_dict["module.prompt_learner.complete_text_embeddings"]

new_state_dict = OrderedDict()
for k, v in load_state_dict.items():
	name = k[7:] # remove `module.`
	new_state_dict[name] = v

vmodel.load_state_dict(new_state_dict, strict=False)

###################
# Start streaming #
###################
# init global buffer to collect frames
global_buffer = Queue(32)
# init local buffer to collect metadata from each tracked ID
# COMMENT: choose between limited/unlimited number of local buffers
#          limited to 8 local buffers for now
local_buffers = []
for i in range(20):
	local_buffers[i] = Queue(32)

pipeline.start(config)
with torch.no_grad():
	while True:
		# Wait for color frame
		frames = pipeline.wait_for_frames()
		color_frame = frames.get_color_frame()
		if not color_frame:
			continue

		# Convert images to numpy arrays then to torch tensors in GPU
		# COMMENT: choose a format to collect the frames (numpy array? torch tensor? BGR? RGB?)
		img = np.asanyarray(color_frame.get_data()) #BGR?? RGB?? Double check if channels are correct
		global_buffer.put(img)
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

		# BYTETracker
		if outputs[0] is not None:
			output = outputs[0].cpu()
			output = output[output[:, 6] == 0] # filter out ONLY persons

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
					local_buffers[tid] = tlwh # push to local buffers here
				
			# Plot BYTETracker output
			# print(online_tlwhs) use this to get scrops for tracked object
			# COMMENT: marked for removal, collect metadata instead
			#vis_res = plot_tracking(img_ori, online_tlwhs, online_ids)
		else:
			#vis_res = img_ori
			
		# COMENT: add local buffer collection and processing here

		# Only show images if processing is completed on buffer
		# COMMENT: show images sequentially and flush all buffers
		if global_buffer.full():
			cv2.imshow('RealSense', vis_res)
			global_buffer.clear()
			for local_buffer in local_buffers:
				local_buffer.clear()
		if cv2.waitKey(1) & 0xff == ord('q'):
			break

pipeline.stop()
