import numpy as np
import os

dirs = os.listdir('dataset_word/landmarks/final')
for d in dirs[:1]:
    path = os.path.join('dataset_word/landmarks/final', d)
    files = os.listdir(path)
    if files:
        filepath = os.path.join(path, files[0])
        data = np.load(filepath)
        print(f"Sample file {filepath}: shape={data.shape}")
        break
