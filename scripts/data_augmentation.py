"""
🧠 Data Augmentation for Sign Language Landmarks
Implements 4 key techniques to reduce confusion matrix off-diagonals
"""

import numpy as np
from typing import Tuple, List


class LandmarkAugmentor:
    """Advanced augmentation for landmark sequences"""
    
    def __init__(self, random_state=42):
        self.rng = np.random.RandomState(random_state)
    
    # ==================== LAYER 1.1: Sliding Windows ====================
    @staticmethod
    def sliding_window_sequences(
        sequence: np.ndarray,
        window_size: int = 30,
        stride: int = 5
    ) -> List[np.ndarray]:
        """
        🔥 MANDATORY: Convert single sequence → multiple overlapping sequences
        
        Input:  Video of 60 frames
        Output: [0-30, 5-35, 10-40, 15-45, ...] windows
        
        Why: Teaches temporal robustness, reduces exact timing memorization
        
        Args:
            sequence: (T, F) - T frames, F features (usually 63 for 21 landmarks × 3)
            window_size: frames per window (30 is sweet spot)
            stride: frames between windows (5 is good)
            
        Returns:
            List of (window_size, F) sequences
        """
        windows = []
        num_frames = sequence.shape[0]
        
        for start in range(0, num_frames - window_size + 1, stride):
            end = start + window_size
            windows.append(sequence[start:end, :])
        
        # If video is shorter than window_size, pad with last frame
        if len(windows) == 0:
            if num_frames < window_size:
                pad_amount = window_size - num_frames
                padded = np.vstack([
                    sequence,
                    np.tile(sequence[-1:, :], (pad_amount, 1))
                ])
                windows.append(padded)
            else:
                windows.append(sequence[:window_size, :])
        
        return windows
    
    # ==================== LAYER 1.2: Time Warping ====================
    def time_warping_augmentation(
        self,
        sequence: np.ndarray,
        warp_factor_range: Tuple[float, float] = (0.7, 1.3)
    ) -> np.ndarray:
        """
        🔥 VERY IMPORTANT: Randomly distort motion speed
        
        Prevents model from relying on exact timing
        Directly reduces confusion between similar gestures
        
        Techniques:
        - Random frame skipping (speed up)
        - Linear interpolation (slow down)
        - Non-linear warping
        
        Args:
            sequence: (T, F) landmark sequence
            warp_factor_range: (min_factor, max_factor)
                              < 1.0 = slow down
                              > 1.0 = speed up
        
        Returns:
            Warped sequence (same or adjusted length)
        """
        num_frames = sequence.shape[0]
        warp_factor = self.rng.uniform(*warp_factor_range)
        
        # Create warped time indices
        original_indices = np.arange(num_frames)
        warped_indices = original_indices * warp_factor
        warped_indices = np.clip(warped_indices, 0, num_frames - 1)
        
        # Interpolate landmarks at warped timepoints
        warped_sequence = np.zeros_like(sequence)
        for feature_idx in range(sequence.shape[1]):
            warped_sequence[:, feature_idx] = np.interp(
                original_indices,
                warped_indices,
                sequence[:, feature_idx]
            )
        
        return warped_sequence
    
    # ==================== LAYER 1.3: Landmark Space Noise ====================
    def add_landmark_noise(
        self,
        sequence: np.ndarray,
        noise_std: float = 0.01,
        noise_type: str = "gaussian"
    ) -> np.ndarray:
        """
        🔥 IMPORTANT: Add realistic noise to coordinates
        
        Why:
        - Simulates natural variation between signers
        - Prevents exact trajectory memorization
        - Improves generalization
        
        Args:
            sequence: (T, F) landmark sequence
            noise_std: standard deviation of noise (0.01 is reasonable)
            noise_type: "gaussian" or "uniform"
        
        Returns:
            Noisy sequence (same shape)
        """
        if noise_type == "gaussian":
            noise = self.rng.normal(0, noise_std, sequence.shape)
        elif noise_type == "uniform":
            noise = self.rng.uniform(-noise_std, noise_std, sequence.shape)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return sequence + noise
    
    # ==================== LAYER 1.4: Motion-based Features ====================
    @staticmethod
    def extract_motion_features(sequence: np.ndarray) -> np.ndarray:
        """
        🔥 CRITICAL: Convert absolute position → motion dynamics
        
        Instead of: raw hand coordinates
        Use:        frame_t - frame_(t-1) (velocity)
        
        This converts: "what is the hand?" → "how does hand move?"
        
        Args:
            sequence: (T, F) - T frames, F features
        
        Returns:
            Motion sequence: frame differences (first frame duplicated)
        """
        motion = np.zeros_like(sequence)
        motion[0] = sequence[0]  # First frame as reference
        motion[1:] = np.diff(sequence, axis=0)
        
        return motion
    
    # ==================== LAYER 1.5: Normalize by Wrist ====================
    @staticmethod
    def normalize_by_wrist(sequence: np.ndarray) -> np.ndarray:
        """
        🔥 ADVANCED: Remove signer-specific position bias
        
        Hand landmarks = 21 points
        - Point 0 = wrist
        - Rest = finger joints
        
        Normalize all points relative to wrist position
        
        Args:
            sequence: (T, 63) where 63 = 21 points × 3 (x,y,z)
        
        Returns:
            Normalized sequence with wrist at origin
        """
        # Wrist is first 3 values (x, y, z)
        wrist = sequence[:, :3:].reshape(sequence.shape[0], 1, 3)  # (T, 1, 3)
        
        # Reshape sequence into (T, 21, 3)
        reshaped = sequence.reshape(sequence.shape[0], -1, 3)
        
        # Subtract wrist position from all points
        normalized = reshaped - wrist
        
        # Flatten back to (T, 63)
        return normalized.reshape(sequence.shape[0], -1)
    
    # ==================== LAYER 1.6: CRITICAL FIX - Relative Landmark Features ====================
    @staticmethod
    def extract_relative_features(sequence: np.ndarray) -> np.ndarray:
        """
        🔥🔥🔥 FIX 1 - MOST IMPORTANT: Extract relative landmark features
        
        Instead of raw coordinates, compute:
        1. Distances between key joints (finger lengths, palm span)
        2. Angles between fingers
        3. Hand configuration descriptors
        
        This DIRECTLY fixes:
        - hatya vs aapatkal (different hand shapes)
        - aspatal vs prahari (different finger positions)
        - jhagada vs sahayata (different orientations)
        
        Hand landmark structure (21 points):
        - 0: wrist
        - 1-4: thumb (base, mid, pip, tip)
        - 5-8: index (base, mid, pip, tip)
        - 9-12: middle
        - 13-16: ring
        - 17-20: pinky
        
        Args:
            sequence: (T, 63) where 63 = 21 points × 3
        
        Returns:
            (T, feature_count) with raw coords + relative features
        """
        num_frames = sequence.shape[0]
        reshaped = sequence.reshape(num_frames, 21, 3)  # (T, 21, 3)
        
        # Keep original features
        original = sequence.copy()
        
        # ===== KEY DISTANCES (distinguishes hand shape) =====
        # Distance from wrist to each finger tip
        wrist = reshaped[:, 0, :]  # (T, 3)
        
        distances = []
        
        # Finger tips: 4, 8, 12, 16, 20 (thumb, index, middle, ring, pinky)
        finger_tips = [4, 8, 12, 16, 20]
        for tip_idx in finger_tips:
            tip = reshaped[:, tip_idx, :]  # (T, 3)
            dist = np.linalg.norm(tip - wrist, axis=1)  # (T,)
            distances.append(dist)
        
        # ===== INTER-FINGER DISTANCES (distinguishes finger spacing) =====
        # Distance between adjacent finger tips
        for i in range(len(finger_tips) - 1):
            tip1 = reshaped[:, finger_tips[i], :]
            tip2 = reshaped[:, finger_tips[i+1], :]
            dist = np.linalg.norm(tip2 - tip1, axis=1)
            distances.append(dist)
        
        # ===== ANGLES BETWEEN FINGERS =====
        # Angle between consecutive fingers (hand spread)
        angles = []
        for i in range(len(finger_tips) - 1):
            v1 = reshaped[:, finger_tips[i], :] - wrist  # Vector to finger i
            v2 = reshaped[:, finger_tips[i+1], :] - wrist  # Vector to finger i+1
            
            # Compute angle between vectors
            cos_angle = np.sum(v1 * v2, axis=1) / (
                np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8
            )
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            angles.append(angle)
        
        # ===== FINGER CURL (distance along finger length) =====
        # For each finger: distance from base to tip vs sum of joint distances
        finger_bases = [1, 5, 9, 13, 17]  # Thumb, index, middle, ring, pinky
        curls = []
        
        for base_idx, tip_idx in zip(finger_bases, finger_tips):
            base = reshaped[:, base_idx, :]
            tip = reshaped[:, tip_idx, :]
            # Direct distance
            direct_dist = np.linalg.norm(tip - base, axis=1)
            
            # Sum of joint distances (indicator of curl)
            mid_idx = base_idx + 1
            pip_idx = base_idx + 2
            
            seg1 = np.linalg.norm(reshaped[:, mid_idx, :] - base, axis=1)
            seg2 = np.linalg.norm(reshaped[:, pip_idx, :] - reshaped[:, mid_idx, :], axis=1)
            seg3 = np.linalg.norm(tip - reshaped[:, pip_idx, :], axis=1)
            
            sum_segments = seg1 + seg2 + seg3
            curl_ratio = direct_dist / (sum_segments + 1e-8)  # 1.0 = straight, <1.0 = curled
            curls.append(curl_ratio)
        
        # ===== PALM ORIENTATION (hand normal vector) =====
        # Use 3 points to compute palm normal: wrist, middle base, ring base
        wrist_pt = reshaped[:, 0, :]  # (T, 3)
        middle_base = reshaped[:, 9, :]  # (T, 3)
        ring_base = reshaped[:, 13, :]  # (T, 3)
        
        # Vectors in palm plane
        v1_palm = middle_base - wrist_pt
        v2_palm = ring_base - wrist_pt
        
        # Normal to palm (cross product)
        normal = np.cross(v1_palm, v2_palm)
        normal_norm = np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8
        normal_unit = normal / normal_norm  # (T, 3)
        
        # Use normal components as features
        palm_orientation = normal_unit  # (T, 3)
        
        # ===== COMBINE ALL FEATURES =====
        # Stack: original (63) + distances (9) + angles (4) + curls (5) + palm_orientation (3)
        feature_list = [original]
        
        # Add distance features
        for dist in distances:
            feature_list.append(dist[:, np.newaxis])  # (T,) -> (T, 1)
        
        # Add angle features
        for angle in angles:
            feature_list.append(angle[:, np.newaxis])
        
        # Add curl features
        for curl in curls:
            feature_list.append(curl[:, np.newaxis])
        
        # Add palm orientation
        feature_list.append(palm_orientation)
        
        # Concatenate all features
        combined = np.hstack(feature_list)  # (T, 63 + 9 + 4 + 5 + 3) = (T, 84)
        
        return combined
    
    # ==================== COMBINED PIPELINE ====================
    def augment_sequence(
        self,
        sequence: np.ndarray,
        augmentation_config: dict = None
    ) -> List[np.ndarray]:
        """
        🟢 Apply full augmentation pipeline to one sequence
        
        Args:
            sequence: (T, F) landmark sequence
            augmentation_config: dict with keys:
                - apply_relative_features: bool (default True) - FIX 1 ⭐
                - apply_motion: bool (default True)
                - apply_normalization: bool (default True)
                - window_size: int (default 40) - FIX 4 ⭐
                - window_stride: int (default 5)
                - num_warps: int (default 2)
                - noise_std: float (default 0.01)
        
        Returns:
            List of augmented sequences
        """
        if augmentation_config is None:
            augmentation_config = {
                'apply_relative_features': True,  # FIX 1
                'apply_motion': True,
                'apply_normalization': True,
                'window_size': 40,  # FIX 4: Increased from 30 to 40
                'window_stride': 5,
                'num_warps': 2,
                'noise_std': 0.01
            }
        
        augmented = []
        
        # 🔥 FIX 1: Extract relative landmark features FIRST (before other transforms)
        if augmentation_config.get('apply_relative_features', True):
            sequence_with_features = self.extract_relative_features(sequence)
        else:
            sequence_with_features = sequence
        
        # Step 1: Normalize by wrist if requested
        if augmentation_config.get('apply_normalization', True):
            # Note: normalize_by_wrist expects (T, 63) but we might have more features now
            # So we'll apply it to the original 63 features only
            if sequence_with_features.shape[1] > 63:
                # Extract original 63 features, normalize, then re-add computed features
                original_63 = sequence_with_features[:, :63]
                computed_features = sequence_with_features[:, 63:]
                
                original_63_norm = self.normalize_by_wrist(original_63)
                sequence_with_features = np.hstack([original_63_norm, computed_features])
            else:
                sequence_with_features = self.normalize_by_wrist(sequence_with_features)
        
        # Step 2: Extract motion features if requested
        if augmentation_config.get('apply_motion', True):
            sequence_with_features = self.extract_motion_features(sequence_with_features)
        
        # Step 3: Time warping
        num_warps = augmentation_config.get('num_warps', 2)
        for _ in range(num_warps):
            warped = self.time_warping_augmentation(sequence_with_features)
            warped = self.add_landmark_noise(
                warped,
                noise_std=augmentation_config.get('noise_std', 0.01)
            )
            augmented.append(warped)
        
        # Step 4: Original sequence with noise
        noisy = self.add_landmark_noise(
            sequence_with_features,
            noise_std=augmentation_config.get('noise_std', 0.01)
        )
        augmented.append(noisy)
        
        # Step 5: Sliding windows on all variants
        window_size = augmentation_config.get('window_size', 40)  # FIX 4
        window_stride = augmentation_config.get('window_stride', 5)
        
        final_windows = []
        for aug_seq in augmented:
            windows = self.sliding_window_sequences(
                aug_seq,
                window_size=window_size,
                stride=window_stride
            )
            final_windows.extend(windows)
        
        return final_windows


# ==================== TESTING ====================
if __name__ == "__main__":
    print("✅ Augmentation module loaded")
    
    # Test with dummy data
    dummy_seq = np.random.randn(50, 63)  # 50 frames, 63 features
    
    augmentor = LandmarkAugmentor()
    
    # Test sliding windows
    windows = augmentor.sliding_window_sequences(dummy_seq, window_size=30, stride=5)
    print(f"✅ Sliding windows: {len(windows)} sequences of shape {windows[0].shape}")
    
    # Test full pipeline
    augmented = augmentor.augment_sequence(dummy_seq)
    print(f"✅ Full augmentation: {len(augmented)} sequences")
    print(f"   Each shape: {augmented[0].shape}")
