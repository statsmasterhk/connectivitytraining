import copy
import cv2
import numpy as np
import util.config
import torch

### Direct copy from Tracking/Models/ReID.py preprocess function
def preprocess(im, px_mean):

  '''
  Preprocessing the list of images and return a numpy array of preprocessed images.
  Preproccsing involves:
    1) convert the image to float
    2) subtract the pixel mean [104, 117, 123] from the image
    3) Resize the image into size (224, 224, 3)
    4) assign the image to the ndarray
    
  Parameters
  ----------
  im : list of images
    The cap images to be pre-processed
  px_mean : list of 3 elements
    The pixel mean [104, 117, 123]. This mean will be subtracted from the input image to normalise the image
      
  Returns
  -------
  out: numpy array of float32
    The preprocessed images
  '''

  target_size = 224 # the input size of the cap image to the model
  out = np.zeros((len(im), target_size, target_size, 3))

  for i in range(len(im)):
    
    im[i] = im[i].astype(np.float32, copy=False)
    im[i] -= px_mean
    out[i,:,:,:] = cv2.resize(im[i], (target_size,target_size), interpolation=cv2.INTER_LINEAR)
      
  return out

### Direct copy from Tracking/Models/ReID.py get_feature function
def get_feature(model_dir, track, caps):

  '''
  Obtain the feature vectors for the jockey caps with the re-identification network.
  This network generates cap feature vector of length 128 for re-identification purposes. Euclidean distance can be computed between the cap features. The shorter the distance, the more likely the caps belong to the same jockey
  
  Parameters
  ----------
  model_dir : str
  	The directory to the release models, usually on dropbox or company server
  track : str
  	The track name, i.e. 'HVT', 'STT', 'Kranji'
  caps : list of images
  	The cap images
  
  Returns
  -------
  feats : numpy array of float32 
  	Feature vectors of shape 128
  
  Notes
  -----
  
  Post-processing:	Normalisation on the output is necessary for it to be comparible. code 'embed = ip_feat / np.linalg.norm(ip_feat, 2, 1, True)'
  
  Network structure: 	Inception v1 backbone + Triplet Loss
  
  Future Work:		Re-training the model with a more advanced architecture like the one in jockey re-identification and reduce the size of the input
  '''
  
  # Get and load the model from the release model dir
  device = util.config.device
  network_path, px_mean, batch = util.config.get_model_info(model_dir, track, "ReID")
  net = torch.load(network_path).to(device)
  net.eval()
  print("Network Loaded")
  
  # separate the caps into batches, preprocess and extract features batch by batch
  passes = int(np.ceil(len(caps) / batch))
  feats = np.zeros((len(caps), 128))
  for i in range(passes):
    
    if i == passes - 1:
      im = preprocess(caps[i*batch:], px_mean)
    else:
      im = preprocess(caps[i*batch:(i+1)*batch], px_mean)
    im = np.transpose(im, [0,3,1,2]) # transformation to make the channel first
    
    input_frames = torch.tensor(im, dtype=torch.float32).to(device)
    with torch.no_grad(): # disable gradient, necessary here
      blobs = net(input_frames)
    ip_feat = blobs['ip_feat'].data.cpu()
    embed = ip_feat / np.linalg.norm(ip_feat, 2, 1, True) # normalise the features to make them comparible to each other
    
    # load the result into the numpy array for return
    if i == passes - 1:
      feats[i*batch:,:] = copy.deepcopy(embed[:len(caps)-i*batch,:])
    else:
      feats[i*batch:(i+1)*batch,:] = copy.deepcopy(embed)

  return feats

### Direct copy from Tracking/utils/find_trajectories.py box2im function
def box2im(box, image):
  '''obtain the image from the bounding box'''
  return image[box[0]:box[2],box[1]:box[3],:].copy()

### Direct copy from Tracking/utils/find_trajectories.py triplet_feature_to_list function
def triplet_feature_to_list(feat_npy, detected_boxes):
  '''
  convert the structure of the triplet feature into the same as detected_boxes, i.e. list of list of features
  
  Parameters
  ----------
  feat_npy : numpy array
    triplet feature in the shape of (total item number, feature_vector_size). If the detected caps are 10000, then, shape is (10000, 128) If the detected jockeys are 8000, then shape is (8000, 2048).
    Note the difference in feature vector size, as the caps triplet model is still the old one. 
  detected_boxes: list of list of list
    for every frame, for every cap/jockey, there is a bounding box
  Returns
  -------
  feat_list : list of list of vector(numpy array)
      the triplet feature arranged in similar format as detected_boxes
  '''
  feat_list = []
  count = 0
  for frm in range(len(detected_boxes)):
    feat_list.append([])
    for cbid in range(len(detected_boxes[frm])):
      feat_list[-1].append(feat_npy[count,:])
      count += 1
  return feat_list

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx