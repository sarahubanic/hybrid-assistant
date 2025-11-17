"""
Simple visual DB tester.
Usage:
    python tools/test_visual_db.py

It loads `learned_items/visual_db.pkl` and `learned_items/visual_samples/*` and computes ORB descriptors
and matching scores using the same heuristic as the GUI. It prints per-image best-match labels and scores,
so you can see if samples for a label match themselves and how they compare to other labels.
"""
import os
import pickle
import cv2
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), '..')
LEARNED = os.path.join(BASE, 'learned_items')
VIS_SAMPLES = os.path.join(LEARNED, 'visual_samples')
VIS_DB = os.path.join(LEARNED, 'visual_db.pkl')

ORB_FEATURES = 500
DIST_CUTOFF = 60

def load_visual_db():
    if not os.path.exists(VIS_DB):
        print('visual_db.pkl not found at', VIS_DB)
        return {}
    with open(VIS_DB, 'rb') as f:
        return pickle.load(f)


def compute_des(img_path, orb):
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    return des


def match_ratio(des_q, des_db, bf, cutoff=DIST_CUTOFF):
    if des_q is None or des_db is None:
        return 0.0
    try:
        matches = bf.match(des_q, des_db)
    except Exception:
        return 0.0
    good = [m for m in matches if m.distance < cutoff]
    denom = max(1, min(len(des_q), len(des_db)))
    return len(good) / denom


def main():
    orb = cv2.ORB_create(ORB_FEATURES)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    visual_db = load_visual_db()
    if visual_db:
        print('Loaded visual_db with labels:', list(visual_db.keys()))
    else:
        print('visual_db empty or missing; will compute descriptors from samples')

    # Walk samples
    labels = sorted([d for d in os.listdir(VIS_SAMPLES) if os.path.isdir(os.path.join(VIS_SAMPLES, d))])
    if not labels:
        print('No visual_samples found under', VIS_SAMPLES)
        return

    for label in labels:
        folder = os.path.join(VIS_SAMPLES, label)
        imgs = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not imgs:
            continue
        print('\nLabel:', label, 'samples:', len(imgs))
        for img_path in imgs:
            des_q = compute_des(img_path, orb)
            best_label = None
            best_score = 0.0
            # compare against visual_db if available, else against other sample descriptors
            if visual_db:
                for db_label, des_list in visual_db.items():
                    for d in des_list:
                        r = match_ratio(des_q, d, bf)
                        if r > best_score:
                            best_score = r
                            best_label = db_label
            else:
                # fallback: brute-force compare to other sample images
                for other_label in labels:
                    other_folder = os.path.join(VIS_SAMPLES, other_label)
                    for other_img in sorted([os.path.join(other_folder,f) for f in os.listdir(other_folder) if f.lower().endswith(('.png','.jpg'))]):
                        if other_img == img_path:
                            continue
                        des_db = compute_des(other_img, orb)
                        r = match_ratio(des_q, des_db, bf)
                        if r > best_score:
                            best_score = r
                            best_label = other_label

            print(' ', os.path.basename(img_path), '-> best:', best_label, f'score={best_score:.3f}')

if __name__ == '__main__':
    main()
