"""Batch-add visual samples into the learned visual DB.

Usage:
  - Put images into folders under a root, e.g.
      samples/
        toothbrush/
          img1.jpg
          img2.jpg
        cigarette/
          img1.jpg
  - Run from repo root (or give full path):
      python tools/batch_add_visual.py --samples samples --out learned_items/visual_db.pkl

What this does:
  - Computes ORB descriptors for each image and stores them under the object name
    in the visual_db.pkl file compatible with `src/detection_gui.py`'s matching logic.

Notes & advice:
  - For small or texture-poor objects (like cigarette), ORB may struggle. If results
    are poor, consider using a deep embedding (CLIP/DINO) approach (see notes below).
"""

import os
import cv2
import pickle
import argparse


def compute_descriptors_for_image(path, max_features=500):
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(max_features)
    kp, des = orb.detectAndCompute(gray, None)
    return des


def add_folder_samples(samples_root, out_file, max_features=500):
    visual_db = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, 'rb') as f:
                visual_db = pickle.load(f)
        except Exception:
            print(f"Warning: couldn't read existing {out_file}, starting fresh")
            visual_db = {}

    # iterate label folders
    for label in sorted(os.listdir(samples_root)):
        label_dir = os.path.join(samples_root, label)
        if not os.path.isdir(label_dir):
            continue
        print(f"Processing label: {label}")
        des_list = []
        for fname in sorted(os.listdir(label_dir)):
            fpath = os.path.join(label_dir, fname)
            if not os.path.isfile(fpath):
                continue
            des = compute_descriptors_for_image(fpath, max_features=max_features)
            if des is not None:
                des_list.append(des)
                print(f"  - added descriptors from {fname} (shape={des.shape})")
            else:
                print(f"  - skipped {fname} (could not read)")
        if des_list:
            if label in visual_db:
                visual_db[label].extend(des_list)
            else:
                visual_db[label] = des_list

    # write out
    with open(out_file, 'wb') as f:
        pickle.dump(visual_db, f)
    print(f"Saved visual DB to {out_file} with {len(visual_db)} labels")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', '-s', required=True, help='Root folder with label subfolders')
    parser.add_argument('--out', '-o', default='learned_items/visual_db.pkl', help='Output visual DB file')
    parser.add_argument('--features', '-f', type=int, default=500, help='ORB max features')
    args = parser.parse_args()

    if not os.path.isdir(args.samples):
        print('Samples folder not found:', args.samples)
        raise SystemExit(1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    add_folder_samples(args.samples, args.out, max_features=args.features)

    print('\nDone. Next steps:')
    print('- Restart the assistant and press "Describe Scene" or run detection to test matching.')
    print('- If ORB matching is poor for small objects, consider a deep-embedding approach:')
    print('  Use CLIP (clip-vit) to compute image embeddings per sample and match by cosine similarity.')
