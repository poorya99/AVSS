import os
import glob
import subprocess
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import time

import warnings
warnings.filterwarnings("ignore")

class AudioVisualDataset(Dataset):
    def __init__(self, root_dir,  batch_size = 8, sample_rate=16000):

        self.root_dir = root_dir
        # Find all mp4 files recursively
        self.sample_rate = sample_rate
        self.batch_size = batch_size
                # Find all mp4 files recursively
        self.video_paths = glob.glob(os.path.join(root_dir, '**', '*.pt'), recursive=True)
        print(f"Total videos found: {len(self.video_paths)}")

        # Extract durations and create a list of tuples (video_path, duration)
        self.video_duration_list = self._extract_durations()

        # Sort the list based on duration in ascending order
        self.video_duration_list.sort(key=lambda x: x[1])
        # print(self.video_duration_list)

        # Create minibatches where each batch contains videos of the same duration
        self.minibatch = self._create_minibatches()

    def __len__(self):
        return len(self.video_paths)//self.batch_size

    def _extract_durations(self):

        duration_list = []
        for video_path in self.video_paths:
            filename = os.path.basename(video_path)
            # Example filename: B2aQs-P12t4#00081#8492-9025_cropped.mp4
            try:
                # Split the filename to extract start and end
                # Split the string to extract the number
                parts = filename.split('#')  # Split by '#'
                duration = parts[2].split('_')[0]  # Take the 3rd part and split by '_' to isolate '99'
                duration = int(duration)
                if duration <= 0:
                    print(f"Invalid duration in filename: {filename}")
                    continue
                duration_list.append((video_path, duration))
            except Exception as e:
                print(f"Error parsing filename {filename}: {e}")
                continue
        print(f"Total valid videos with extracted durations: {len(duration_list)}")
        return duration_list

    def _create_minibatches(self):
        extracted = [a for a, _ in self.video_duration_list]
        return [extracted[i:i + self.batch_size] for i in range(0, len(extracted), self.batch_size)]


    def _match_lengths(self, audio_features, video_frames):
        # audio_features shape: (channel, n_features, A) - where A is number of audio frames in feature domain
        # video_frames shape: (V, C, H, W) - V is number of video frames

        # We need to clarify what "same number of frames" means:
        # Typically audio features have a certain time resolution (A frames),
        # and video has a certain number of frames (V).
        # We'll assume we want to match them by time indices.
        # If A != V, we clip the longer one.
        
        A = audio_features.shape[0]
        V = video_frames.shape[0]

        min_len = min(A, V)
        # Clip both to min_len
        audio_features = audio_features[:min_len, ...]
        video_frames = video_frames[:min_len, ...]
        return audio_features, video_frames

    def __getitem__(self, idx):
        batch_lst = self.minibatch[idx]


        # Create a temporary directory for extraction
        # In a real scenario, you might use a more permanent cache directory.
        # Here, we assume you have a scratch directory you can use.

        audio_features_batch, video_frames_batch, face_frame_batch, speaker_embedding_batch = [],[],[],[]

        min_len = 99999
        for video_path in batch_lst:
            # Extract audio and features
            data = torch.load(video_path)
            audio_features = data["audio_features"]
            video_frames=data["video_frames"]
            face_frame=data["face_frame"]
            speaker_embedding = data["speaker_embedding"]
            vid_len = audio_features.shape[0]
            if min_len>vid_len:
                min_len = vid_len
            audio_features_batch.append(audio_features); video_frames_batch.append(video_frames)
            face_frame_batch.append(face_frame); speaker_embedding_batch.append(speaker_embedding)
        # print(len(audio_features_batch), len(video_frames_batch), len)
        for m in range(len(batch_lst)):
            audio_features_batch[m] = audio_features_batch[m][:min_len,...]
            video_frames_batch[m] = video_frames_batch[m][:min_len,...]
        audio_features_batch=np.array(audio_features_batch); video_frames_batch = np.array(video_frames_batch)
        face_frame_batch=np.array(face_frame_batch); speaker_embedding_batch=np.array(speaker_embedding_batch)

        # Clean up tmp_dir if needed, or keep for debugging
        # In production, you might remove the temporary data.
        # import shutil
        # shutil.rmtree(tmp_dir)
        audio_features_batch =torch.tensor(audio_features_batch)
        video_frames_batch = torch.tensor(video_frames_batch)
        face_frame_batch = torch.tensor(face_frame_batch)
        speaker_embedding_batch = torch.tensor(speaker_embedding_batch)

        return audio_features_batch, video_frames_batch, face_frame_batch, speaker_embedding_batch


if __name__=="__main__":
  # Example usage:
  dataset = AudioVisualDataset(root_dir=r"/content/vxclb")
  t_l = []
  for i in range (len(dataset)):
      print(f"batch = {i}")
      t0 = time.time()
      a, v, mm, nn = dataset[i]
      t_l.append(time.time()-t0)
      print(a.shape, v.shape, mm.shape, nn.shape)
  print(sum(t_l), sum(t_l)/len(t_l), t_l)
  loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
  # for audio_feats, video_frames, paths in loader:
  #     # Your training code here
