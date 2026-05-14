import numpy as np
import random

class SequenceAugmentor:

    def __init__(self, noise_std=0.01):
        self.noise_std = noise_std

    # --------------------------------------------------
    # 1. Add Gaussian noise
    # --------------------------------------------------
    def add_noise(self, seq):
        noise = np.random.normal(0, self.noise_std, seq.shape)
        return seq + noise

    # --------------------------------------------------
    # 2. Time warping (speed up / slow down)
    # --------------------------------------------------
    def time_warp(self, seq, factor=0.8):
        length = len(seq)
        new_length = int(length * factor)

        indices = np.linspace(0, length - 1, new_length).astype(int)
        warped = seq[indices]

        # pad back
        if new_length < length:
            pad = np.repeat(warped[-1:], length - new_length, axis=0)
            warped = np.vstack([warped, pad])

        return warped[:length]

    # --------------------------------------------------
    # 3. Frame dropping
    # --------------------------------------------------
    def frame_dropout(self, seq, drop_prob=0.1):
        mask = np.random.rand(len(seq)) > drop_prob
        seq = seq[mask]

        if len(seq) == 0:
            return np.zeros_like(seq)

        # restore length
        while len(seq) < 40:
            seq = np.vstack([seq, seq[-1]])

        return seq[:40]

    # --------------------------------------------------
    # 4. Temporal crop (focus on motion)
    # --------------------------------------------------
    def temporal_crop(self, seq):
        start = random.randint(0, len(seq)//4)
        end = start + int(len(seq)*0.75)
        cropped = seq[start:end]

        while len(cropped) < 40:
            cropped = np.vstack([cropped, cropped[-1]])

        return cropped[:40]

    # --------------------------------------------------
    # 5. Spatial scaling (simulate distance)
    # --------------------------------------------------
    def spatial_scale(self, seq, scale=1.1):
        return seq * scale

    # --------------------------------------------------
    # MAIN AUGMENT FUNCTION
    # --------------------------------------------------
    def augment(self, seq):

        augmented = []

        # original
        augmented.append(seq)

        # noise
        augmented.append(self.add_noise(seq))

        # time warp
        augmented.append(self.time_warp(seq, 0.8))
        augmented.append(self.time_warp(seq, 1.2))

        # frame drop
        augmented.append(self.frame_dropout(seq))

        # crop
        augmented.append(self.temporal_crop(seq))

        # scale
        augmented.append(self.spatial_scale(seq, 0.9))
        augmented.append(self.spatial_scale(seq, 1.1))

        return augmented