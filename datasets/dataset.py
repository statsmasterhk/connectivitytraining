import os
import math 
import torch

import numpy as np

from torch.utils.data import Dataset
from util.result import import_subsequences_results
from util.feature import triplet_feature_to_list

class ConnDataset(Dataset):
	def _get_races(self, races_file_path):
		""" Get to process races from a .txt file, where each line = one racelabel
		
		Args:
		    races_file_path (String): Path to races.txt file
		
		Returns:
		    races: List of races string
		"""
		# Read races_file
		if not os.path.exists(races_file_path):
			print(f"Could not find races_file: {races_file_path}, Please check the file path...")

		races = []
		with open(races_file_path, "r")as f:
			for racelabel in f.readlines():
				racelabel_str = racelabel.strip()
				if racelabel_str:
					races += [racelabel_str]
		return races

	def _get_caps_pair_info(self, cap_0_info, cap_0_feat, cap_1_info, cap_1_feat, t):

		cap_0_cx = int((cap_0_info[1] + cap_0_info[3]) / 2)
		cap_0_cy = int((cap_0_info[0] + cap_0_info[2]) / 2)

		cap_1_cx = int((cap_1_info[1] + cap_1_info[3]) / 2)
		cap_1_cy = int((cap_1_info[0] + cap_1_info[2]) / 2)
		cap_1_w = int((cap_1_info[3] - cap_1_info[1]))
		cap_1_h = int((cap_1_info[2] - cap_1_info[0]))

		feat_diff = np.sum((cap_0_feat - cap_1_feat) ** 2) ** 0.5
		dx = abs(cap_0_cx - cap_1_cx) / (math.sqrt(cap_1_w * cap_1_h) * 100)
		dy = abs(cap_0_cy - cap_1_cy) / (math.sqrt(cap_1_w * cap_1_h) * 50)
		norm_t = t / 2
		return [feat_diff, dx, dy, norm_t]

	def _generate_pairs(self, feat_dir, jkcp_gt_dir):
		# For WeightedRandomSampler, maximum 2**24 samples is supported, we will be subsample
		# from each race if it exceeds the maximum
		max_sample_per_race = int(2**24 / len(self.races))
		print(f"Total {len(self.races)} races to be processed.")
		# Read races_file
		for i in range(len(self.races)):
			each_race = self.races[i]
			print(f"Processing {each_race} ({i+1} / {len(self.races)})...")
			# Read jkcps_gt_flie, skip the race if jkcps_gt_flie doesn't exist
			# For the CSV
			jkcp_gt_path = os.path.join(jkcp_gt_dir, f"{each_race}.csv")
			if not os.path.exists(jkcp_gt_path):
				print(f"Missing jkcp gt file for {each_race}, Skipping...")
				continue
			#print(f"Reading GT boxes...")
			start_frm, end_frm, detected_boxes = import_subsequences_results(jkcp_gt_path)
			#print(f"Num of frames: {len(detected_boxes)}, Start: {start_frm}, End: {end_frm}")

			# Read feature, skip the race if feature file doesn't exist
			feature_path = os.path.join(feat_dir, f"{each_race}.npy")
			if not os.path.exists(feature_path):
				print(f"Missing feature file for {each_race}, Skipping...")
				continue

			feature_npy_path = os.path.join(feat_dir, f"{each_race}.npy")
			triplet_features_caps_npy = np.load(feature_npy_path, "r", True)
			triplet_features_caps = triplet_feature_to_list(triplet_features_caps_npy, detected_boxes)

			# Loop through each t + (1,2,3,4)
			tmp_inputs = []
			tmp_lb_classes = []
			tmp_lbs = []
			for t in range(1, self.max_t + 1):
				#print(f"Processing t={t}...")
				num_of_frm = end_frm - start_frm + 1
				for frm_id in range(num_of_frm):
					next_frm_id = frm_id + t
					# if next_frm_id out of bound/
					if next_frm_id >= num_of_frm:
						continue
					num_caps_frm_0 = len(detected_boxes[frm_id]) # num of caps in frame frm_id
					num_caps_frm_1 = len(detected_boxes[next_frm_id])# num of caps in frame frm_id + t
					for i in range(num_caps_frm_0):
						cap_0_box_info = detected_boxes[frm_id][i] # Contains y1, x1, y2, x2, id
						cap_0_feat = triplet_features_caps[frm_id][i]
						cap_0_id = cap_0_box_info[-1]
						for j in range(num_caps_frm_1):
							cap_1_box_info = detected_boxes[next_frm_id][j] # Contains y1, x1, y2, x2, id
							cap_1_feat = triplet_features_caps[next_frm_id][j]
							cap_1_id = cap_1_box_info[-1]
							inputs = self._get_caps_pair_info(cap_0_box_info, cap_0_feat, cap_1_box_info, cap_1_feat, t)
							# Positive pair if same id, negative pair if not same id
							tmp_inputs.append(inputs)
							if cap_0_id == cap_1_id:
								tmp_lb_classes.append(1)
								tmp_lbs.append([1])
							else:
								tmp_lb_classes.append(0)
								tmp_lbs.append([0])

			if len(tmp_inputs) > max_sample_per_race:
				sub_samp_ids = np.random.choice(len(tmp_inputs), max_sample_per_race)
				for sub_id in sub_samp_ids:
					self.inputs.append(tmp_inputs[sub_id])
					self.label_classes.append(tmp_lb_classes[sub_id])
					self.labels.append(tmp_lbs[sub_id])
				print(f"This Race exceeds max sample per race: {max_sample_per_race} will be randomly sampled")
			else:
				self.inputs.extend(tmp_inputs)
				self.label_classes.extend(tmp_lb_classes)
				self.labels.extend(tmp_lbs)

	def __init__(self, races_file, feat_dir, jkcp_gt_dir):
		# 
		#self.jkcps = # id, x, y, w, h, feature(by running triplet feature extraction)

		self.max_t = 4 # t + (1,2,3,4) frames
		self.classes = 2

		self.feat_dir = feat_dir
		self.jkcp_gt_dir = jkcp_gt_dir

		self.races = self._get_races(races_file) # txt file, containing racelabels of the interested race

		### Seperate positive samples and negative samples?
		self.inputs = []
		self.labels = []
		self.label_classes = []
		self._generate_pairs(feat_dir, jkcp_gt_dir)

	def __len__(self):
		return len(self.inputs)


	def __getitem__(self, idx):
		# return a pos pair and a neg pair so the training samples is more evenly distributed?
		return torch.FloatTensor(self.inputs[idx]), torch.LongTensor(self.labels[idx])


		