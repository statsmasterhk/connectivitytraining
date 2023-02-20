""" An modified copy from config.py, for using some of the functions from the main program,
    Better practice to turn stuff input packages to be called from it directly?
    # To BE DOCUMENTED


Attributes:
    accepted_formats (list): Description
    camerachange_save_path (str): Description
    debug (bool): Description
    dependencies (TYPE): Description
    device (TYPE): Description
    directories (TYPE): Description
    end_max (int): Description
    end_min (int): Description
    feature_size (int): Description
    frame_height (int): Description
    frame_width (int): Description
    gcn_params (TYPE): Description
    max_framediff (int): Description
    offset (int): Description
    pixelstep (int): Description
    segmentation_model_dir (str): Description
    start_max (int): Description
    start_min (int): Description
    tracking (TYPE): Description
    video_processer_save_path (str): Description
"""
import os, json
import torch

# mode
debug = True
device = 'cuda' if torch.cuda.is_available() else 'cpu' # use GPU instead of cpu, cpu not recommended

# task list
class Task:
  FRAMEEXTRACTOR 	= 	-1
  SCENECLASSIFY 	= 	0
  RAILMASK 		= 	1
  RAILPOLEMASK 	= 	2
  SEMANTICMASK 	= 	3
  OPTICALFLOW 		= 	4
  DETECTION 		= 	5
  TRACKING 		= 	6

# a task, i.e. 7, will depend on itself and a list of other tasks, i.e. 0, 1, 6, that generates the necessary data for task 7
# Task 7 is Tracking, which depends on Start End Frame Detection, Camera Change Detection and Jockey/Cap/Saddlecloth Detection
dependencies = {Task.FRAMEEXTRACTOR	: 	[Task.SCENECLASSIFY, Task.FRAMEEXTRACTOR],
                Task.SCENECLASSIFY	: 	[Task.SCENECLASSIFY], 
                Task.RAILMASK		: 	[Task.SCENECLASSIFY, Task.RAILMASK],
                Task.RAILPOLEMASK	: 	[Task.SCENECLASSIFY, Task.RAILPOLEMASK],
                Task.SEMANTICMASK	: 	[Task.SCENECLASSIFY, Task.SEMANTICMASK],
                Task.OPTICALFLOW	: 	[Task.SCENECLASSIFY, Task.RAILMASK, Task.RAILPOLEMASK, Task.SEMANTICMASK, Task.OPTICALFLOW],
                Task.DETECTION	: 	[Task.SCENECLASSIFY, Task.DETECTION],
                Task.TRACKING		: 	[Task.SCENECLASSIFY, Task.DETECTION, Task.TRACKING]
               }

# save directory, all the directories to be created. Please modify if you want to include more directories for saving data
directories = {
	Task.FRAMEEXTRACTOR : {'ROOT': 'Saved_Frames'},
	Task.SCENECLASSIFY : {},
	Task.RAILMASK : {'ROOT': 'Rail_Masks'},
	Task.RAILPOLEMASK : {'ROOT': 'Rail_Pole_Masks'},
	Task.SEMANTICMASK : {'ROOT': 'Semantic_Masks'},
	Task.OPTICALFLOW : {'ROOT': 'Optical_Flow'},
	Task.DETECTION : {'ROOT': 'Detect'},
	Task.TRACKING : {
		'ROOT': 		'Track_Cap', 						# SubTaskTracking.ROOT
    'SADDLECLOTHNUMBER':  os.path.join('Track_Cap','saddlecloth_with_id'),  # SubTaskTracking.SADDLECLOTHNUMBER
		'MATCHING': 	os.path.join('Track_Cap','cap_saddlecloth_match'), 	# SubTaskTracking.MATCHING
		'CONNECTIVITY': 	os.path.join('Track_Cap','conn_feat'), 		# SubTaskTracking.CONNECTIVITY
		'CAPTRIPLET':		os.path.join('Track_Cap','trip_feat'), 		# SubTaskTracking.CAPTRIPLET
		'JOCKEYTRIPLET': 	os.path.join('Track_Cap','trip_feat_jockey'), 	# SubTaskTracking.JOCKEYTRIPLET
		'SUBSEQUENCE': 	os.path.join('Track_Cap','subsequences'), 		# SubTaskTracking.SUBSEQUENCE
		'FINALRESULT': 	os.path.join('Track_Cap','final_result')		# SubTaskTracking.FINALRESULT
		},
	}

# files storage
video_processer_save_path = 'vid_process.csv'
camerachange_save_path = 'camera_change.csv'
#scene_classification_save_path = 'scene_classification.csv'

# video
accepted_formats = ['.mp4', '.mpg']

# Start End Frame - assume race start within 40 seconds of race
start_min	=	0  # min frame number for start frame
start_max 	= 	40 # 40 * 25 fps, max frame number for start frame
offset 	=	20 # offset, the prediction - offset is the true start frame
end_min 	= 	55 # use for calculate the minimum end frame. int(config.end_min * distance * fps / 1000 + start_frame)
end_max 	= 	65 # use for calculate the maximum end frame. int(config.end_max * distance * fps / 1000 + start_frame)

# OpticalFlow
feature_size 	= 	256
frame_width 	= 	1920
frame_height 	= 	1080
pixelstep 	= 	5
max_framediff 	= 	1
segmentation_model_dir="Segmentation-v1.0.0" # directory for the segmentation model


# Tracking
class HVTConfig:
  conn_thres = 0.85
  scn_max = 12

class STTConfig:
  conn_thres = 0.85
  scn_max = 14
  
class STAWTConfig:
  conn_thres = 0.85
  scn_max = 14
  
class KranjiConfig:
  conn_thres = 0.75
  scn_max = 16

tracking = {
  'cap_thres': 0.9,  # read cap detection result threshold
  'saddlecloth_thres': 0.9,  # read saddlecloth detection result threshold
  'jockey_thres': 0.9,  # read jockey detection result threshold
  'matching_thres': 0.9,  # cap_saddlecloth matching threshold
  'match_thres': 0.4,   # these are thresholds for matching in several steps
  'triplet_thres': 0.5,
  'triplet_thres_high': 0.55,
  'triplet_thres_low': 0.35,
  'triplet_thres_jockey': 0.65,
  'triplet_thres_gp': 0.6,
  'overlap_thres': 10
}

tracking['HVT'] = HVTConfig()
tracking['STT'] = STTConfig()
tracking['STAWT'] = STAWTConfig()
tracking['Kranji'] = KranjiConfig()
# Detection

# GCN clustering
gcn_params = {
    'feat_agg_path': 'Clustering/FeatureAggregation/feat_agg_13656.9676_weight.pth',
    'gcn_path': 'Clustering/GCN/epoch_47_weight.pth',
    'f_dim': 512,  # Feature dimension of triplet model output
    'k1': 8,  # Number of 1-hop neighbors
    'k_at_hop': [8,5],  # Number of neighbors at different hops
    'active_connection':5,  # Number of nearest neighbors for constructing the Instance Pivot Subgraph
    'inp_size': 64,  # Frame will be resized to this size before passing into the triplet model
    'dist_metric': 'l2',  # Distance metric to use for the triplet model
    'device': device,
    'thres': 0.5,  # Minimum matching threshold
    'batch_size': 8,
    'n_workers': 6, #os.cpu_count()-4,
    'pin_memory': False
}


# Models
def get_model_paths(task, model_type):

  models = {
    "Siamese": {
      "network": f"Tracking-v2.2/InitialTracking/Siamese_CNN_iter_10000_{model_type}_model.pth",
      "mean": "",
      "batch": 1000
    },
    "Conn": {
      "network": f"Tracking-v2.2/InitialTracking/Conn_CNN_iter_10000_{model_type}_model.pth",
      "mean": "",
      "batch": -1
    }, 
    "ReID": {
      "network": f"Tracking-v2.2/Reidentification/{model_type}/triplet_{model_type.lower()}_model.pth",
      "mean": [104, 117, 123],
      "batch": 50
    },
    "ReID_Jockey": {
      "network": f"Tracking-v2.2/ReID_Jockey/{model_type}.pth.tar",
      "mean": {'px_mean': [0.485, 0.456, 0.406],'px_std': [0.229, 0.224, 0.225], 'input_size': {'Kranji': 128, "HVT": 256, "STT": 256,"STAWT":256}, 'num_classes': {'Kranji': 289, 'HVT': 160, 'STT': 160, 'STAWT': 160}},
      "batch": 50
    },
    "ReID_SCN": {
      "network": f"Tracking-v2.2/NumberDetection/model_scn_{model_type}",
      "mean": "",
      "batch": 1000
    },
    "Detection": {
      "network": f"Detection-v2.1.0/{model_type}_jockey_cap_sc.yaml",
      "mean": "",
      "batch": 1
    },
    "SceneClassify": {
      ### Without sudden 20220121 model
      "decoder": "SceneClassification/decoder-ep-9-0.9944798886941024.pth",
      "encoder": "SceneClassification/encoder-ep-9-0.9944798886941024.pth",
      # ### Without sudden
      # "decoder": "SceneClassification/decoder-ep-41-0.9973659136519434.pth",
      # "encoder": "SceneClassification/encoder-ep-41-0.9973659136519434.pth",
      ### With sudden
      # "decoder": "SceneClassification/decoder-ep-42-0.9976996234442629.pth",
      # "encoder": "SceneClassification/encoder-ep-42-0.9976996234442629.pth",
    },
    "RailSegmentation": {
      "network": "Rail-v2.1.0/Rail_Tensorflow/rail_model",
      "mean": "",
      "batch": 1,
    },
    "RailPoleSegmentation": {
      "network": "Rail-v2.1.0/Rail_Pole_Pytorch/rail_pole_model_final.pth",
      "mean": "",
      "batch": 1,
    },
    "SemanticSegmentation": {
      "network": "Segmentation-v1.0.0/All_Objects/highest_model",
      "mean": "",
      "batch": 1,
    },
    "DDFlow": {
      "network": "DDFlow/data_distillation",
      "mean": "",
      "batch": 1,
    }
  }
  
  return models[task]


def get_model_info(model_dir, track, task, distance=None, course=None):

  if task == "Conn":
    if track == "Kranji":
      if course == "P" and distance <= 1200 or course == "S" and distance <= 1600 or course == "L" and distance <= 1800:
        model_type = "Kranji-short"
      else:
        model_type = "Kranji-long"
    else:
      model_type = track
  elif task in ["ReID", "Detect_CAP", "Detect_SCN", "ReID_Jockey"]:
    if track in ["HVT", "STT", "STAWT"]:
      model_type = "HongKong"
    elif track in ["Kranji"]:
      model_type = "Singapore"
  else:
    model_type = track

  model_info = get_model_paths(task, model_type)
  network, mean_path, batch = model_info["network"], model_info["mean"], model_info["batch"]
  network_path = os.path.join(model_dir, network)
  
  return network_path, mean_path, batch

