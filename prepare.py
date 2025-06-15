import zipfile
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, deque
import ace_tools as tools

# Unzip dataset if not already (to /mnt/data)
ZIP_PATH = '/mnt/data/arc-prize-2025.zip'
DATA_DIR = '/mnt/data/arc-prize-2025'
TRAIN_CH_PATH = os.path.join(DATA_DIR, 'arc-agi_training_challenges.json')

if not os.path.isfile(TRAIN_CH_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall('/mnt/data')

# Load JSON data
with open(TRAIN_CH_PATH, 'r') as f:
    train_ch = json.load(f)

# Normalize to dict id->challenge
if isinstance(train_ch, list):
    train_map = {c['id']: c for c in train_ch}
else:
    train_map = train_ch

# Feature extraction
def extract_features(ch):
    if not ch.get('train'):
        return None
    grid = np.array(ch['train'][0]['input'])
    h, w = grid.shape
    hist = Counter(grid.flatten())
    mask = (grid != 0).astype(int)
    sym_h = int(np.array_equal(mask, mask[::-1, :]))
    sym_v = int(np.array_equal(mask, mask[:, ::-1]))
    sym_d = int(h == w and np.array_equal(mask, mask.T))
    # Components
    visited = np.zeros_like(mask, dtype=bool)
    comps = 0
    def neighbors(r, c):
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w:
                yield nr, nc
    for i in range(h):
        for j in range(w):
            if mask[i,j] and not visited[i,j]:
                comps += 1
                queue = deque([(i,j)])
                visited[i,j] = True
                while queue:
                    r, c = queue.popleft()
                    for nr, nc in neighbors(r,c):
                        if mask[nr,nc] and not visited[nr,nc]:
                            visited[nr,nc] = True
                            queue.append((nr,nc))
    return {
        'height': h,
        'width': w,
        'num_colors': len(hist),
        'max_freq': max(hist.values()),
        'sym_h': sym_h,
        'sym_v': sym_v,
        'sym_d': sym_d,
        'components': comps,
        'train_examples': len(ch.get('train', [])),
        'test_examples': len(ch.get('test', []))
    }

# Build DataFrame
records = []
for cid, ch in train_map.items():
    feats = extract_features(ch)
    if feats is not None:
        feats['id'] = cid
        records.append(feats)
df = pd.DataFrame(records)

# Display DataFrame
tools.display_dataframe_to_user("ARC-Prize 2025 Train Puzzle Features", df)

# Plot distributions
plt.figure()
plt.hist(df['height'], bins=20)
plt.title('Grid Height Distribution')
plt.xlabel('Height')
plt.ylabel('Count')
plt.show()

plt.figure()
plt.hist(df['num_colors'], bins=range(df['num_colors'].max()+2))
plt.title('Number of Colors Distribution')
plt.xlabel('Num Colors')
plt.ylabel('Count')
plt.show()

plt.figure()
plt.hist(df['components'], bins=20)
plt.title('Connected Component Count Distribution')
plt.xlabel('Components')
plt.ylabel('Count')
plt.show()
