import numpy as np
from collections import OrderedDict
from queue import Queue
import argparse
import cv2
import torch

#from multiprocessing import Pool

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

parser = argparse.ArgumentParser(description ='Person detection, tracking and action recogniton.')
parser.add_argument("-v", "--videopath",
					type = str,
					help ='path to video you wish to perform inference on')
parser.add_argument("-ht", "--height",
					type = int,
					help ='height of video')
parser.add_argument("-wt", "--width",
					type = int,
					help ='width of video')
args = parser.parse_args()

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

########################
# Configure cv2 stream #
########################
cap = cv2.VideoCapture(args.videopath)
width, height = args.width, args.height

out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (width, height))

####################
# load BYTETracker #
####################
aspect_ratio_thresh = 0.6
min_box_area = 10
track_thresh = 0.5
track_buffer = 60
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
class_names = ['walk','jump','run']

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
'''
for param in vmodel.parameters():
	print(param.dtype)
'''

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

#######################################
# utility functions (to be offloaded) #
#######################################
def extract_tracks(global_buffer, local_buffers):
	vid = np.array(global_buffer.queue)
	tidlst, bboxlst = zip(*local_buffers.items())
	def extract_crop(bbox):
		tlx, tly, brx, bry = bbox.astype(int)
		crop_vid = vid[:,tly:bry,tlx:brx,:]
		return crop_vid
	
	cropped_clips = []
	for i in range(len(tidlst)):
		tmp = np.expand_dims(process_vid(extract_crop(bboxlst[i])),0)
		cropped_clips.append(tmp)
	
	cropped_clips = np.concatenate(cropped_clips,axis = 0)
	cropped_clips = torch.HalfTensor(cropped_clips) # B, n, c, h, w
	return cropped_clips
	
def process_vid(cropped_clip):
	scale = 224
	normalized = normalize_vid(cropped_clip)
	resized = resize_vid(normalized, scale)
	cropped = centerCrop_vid(resized)
	cropped = nhwc_to_nchw(cropped)
	return cropped
	
def normalize_vid(cropped_clip, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
	R, G, B = cropped_clip[:,:,:,0], cropped_clip[:,:,:,1], cropped_clip[:,:,:,2]
	R, G, B = (R - mean[0])/std[0], (G - mean[1])/std[1], (R - mean[2])/std[2]
	R, G, B = np.expand_dims(R, 3), np.expand_dims(G, 3), np.expand_dims(B, 3)
	return np.concatenate((R,G,B),axis=3)
	
def resize_vid(cropped_clip, scale):
	n, h, w, _ = cropped_clip.shape
	if h < w:
		new_h, new_w = scale, int(w* scale / h)
	else:
		new_h, new_w = int(h* scale / w), scale
	
	resized_clip = np.array([cv2.resize(cropped_clip[i],(new_w, new_h)) for i in range(n)])
	return resized_clip
	
def centerCrop_vid(cropped_clip):
	_, h, w, _ = cropped_clip.shape
	diff = abs(int((h - w)/2))
	if h < w:
		centercropped_clip = cropped_clip[:,:,diff:diff+h,:]
	else:
		centercropped_clip = cropped_clip[:,diff:diff+w,:,:]
	return centercropped_clip

def nhwc_to_nchw(np_clip):
	return np.transpose(np_clip, (0,3,1,2))
	
###################
# Start streaming #
###################
# init global buffer to collect frames
global_buffer = Queue(32)
# init local buffer to collect metadata from each tracked ID
local_buffers = dict()

with torch.no_grad():
	while True:
		# Wait for color frame
		ret, img = cap.read() 
		if not ret:
			break

		# BGR?? RGB?? Double check if channels are correct
		#global_buffer.put(img)

		# inference
		img_ori = img
		ratio = min(test_size[0] / height, test_size[1] / width)
		img, _ = preproc(img, None, test_size)
		img = torch.from_numpy(img).unsqueeze(0).float().half().cuda()
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
					tlbr = tlwh + np.array([0, 0, tlwh[0], tlwh[1]])
					# update local buffers with metadata
					if tid not in local_buffers:
						local_buffers[tid] = tlbr
					else:
						old_tlbr = local_buffers.get(tid)
						'''
						tmpmsk = old_tlbr < tlbr
						local_buffers[tid][tmpmsk] = tlbr[tmpmsk]
						'''
						if tlbr[0] < old_tlbr[0]:
							old_tlbr[0] = tlbr[0]
						if tlbr[1] < old_tlbr[1]:
							old_tlbr[1] = tlbr[1]
						if tlbr[2] > old_tlbr[2]:
							old_tlbr[2] = tlbr[2]
						if tlbr[3] > old_tlbr[3]:
							old_tlbr[3] = tlbr[3]
						local_buffers[tid] = old_tlbr
					vis_res = plot_tracking(img_ori, online_tlwhs, online_ids)
		else:
			vis_res = img_ori
		global_buffer.put(vis_res)

		if global_buffer.full():
			cropped_clips = extract_tracks(global_buffer, local_buffers).cuda()
			
			
			with torch.cuda.amp.autocast():
				logits = vmodel(cropped_clips)
				pred = logits.argmax(1)
				pred_actions = [class_names[i] for i in pred]
				print(pred_actions)
			
			
			while not global_buffer.empty():
				tmp = global_buffer.get()
				
				i = 0
				for bbox in local_buffers.values():
					tlx, tly, brx, bry = bbox.astype(int)
					cv2.rectangle(tmp, (tlx, tly), (brx, bry), (36,255,12), 1)
					cv2.putText(tmp, pred_actions[i], (tlx, tly-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
					i += 1
				
				cv2.imshow('clip', tmp)
				out.write(tmp)
				cv2.waitKey(1)
			
			local_buffers.clear()
			
		if cv2.waitKey(1) & 0xff == ord('q'):
			break
cap.release()
out.release()
cv2.destroyAllWindows() 
