import json
import math
import os
from typing import List, Dict, Tuple
import numpy as np

Minutia = Dict[str, float]

def load_minutiae(file_path: str) -> List[Minutia]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data if data else []

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def rotate_point(x: float, y: float, angle: float, origin=(0,0)):
    ox, oy = origin
    x -= ox
    y -= oy
    xr = x * math.cos(angle) - y * math.sin(angle)
    yr = x * math.sin(angle) + y * math.cos(angle)
    return xr + ox, yr + oy

def align_minutiae(minutiae: List[Minutia], angle: float, translation: Tuple[float,float]):
    tx, ty = translation
    return [
        {**m, 'x': rotate_point(m['x'], m['y'], angle)[0]+tx,
             'y': rotate_point(m['x'], m['y'], angle)[1]+ty}
        for m in minutiae
    ]

def compute_centroid(minutiae: List[Minutia]) -> Tuple[float, float]:
    if not minutiae:
        return 0.0, 0.0
    xs = [m['x'] for m in minutiae]
    ys = [m['y'] for m in minutiae]
    return np.mean(xs), np.mean(ys)

def match_minutiae(minutiae1: List[Minutia], minutiae2: List[Minutia],
                    dist_thresh=15.0, angle_thresh=0.3) -> float:
    if not minutiae1 or not minutiae2:
        return 0.0

    used = set()
    score = 0.0
    total_weight = sum(m['quality']*m['coherence'] for m in minutiae1) or 1.0

    for m1 in minutiae1:
        best_score = 0.0
        best_idx = -1
        for idx, m2 in enumerate(minutiae2):
            if idx in used:
                continue
            dist = euclidean_distance((m1['x'], m1['y']), (m2['x'], m2['y']))
            angle_diff = abs(m1['orientation'] - m2['orientation'])
            if dist <= dist_thresh and angle_diff <= angle_thresh and m1['type'] == m2['type']:
                match_score = m1['quality']*m1['coherence']*m2['quality']*m2['coherence']
                if match_score > best_score:
                    best_score = match_score
                    best_idx = idx
        if best_idx >= 0:
            score += best_score
            used.add(best_idx)

    return score / total_weight

def match_two_users(user1_path: str, user2_path: str) -> float:
    m1 = load_minutiae(user1_path)
    m2 = load_minutiae(user2_path)
    if not m1 or not m2:
        return 0.0

    # centratura
    c1 = compute_centroid(m1)
    c2 = compute_centroid(m2)
    translation = (c1[0]-c2[0], c1[1]-c2[1])

    # orientamento medio
    angle1 = np.mean([m['orientation'] for m in m1]) if m1 else 0.0
    angle2 = np.mean([m['orientation'] for m in m2]) if m2 else 0.0
    rotation = angle1 - angle2

    m2_aligned = align_minutiae(m2, rotation, translation)
    return match_minutiae(m1, m2_aligned)

def batch_match(features_dir: str) -> Dict[Tuple[str, str], float]:
    results = {}
    users = [d for d in os.listdir(features_dir) if os.path.isdir(os.path.join(features_dir,d))]
    for i, u1 in enumerate(users):
        for j, u2 in enumerate(users):
            if j <= i:
                continue
            u1_files = [f for f in os.listdir(os.path.join(features_dir,u1)) if f.endswith('_minutiae.json')]
            u2_files = [f for f in os.listdir(os.path.join(features_dir,u2)) if f.endswith('_minutiae.json')]
            max_score = 0.0
            for f1 in u1_files:
                for f2 in u2_files:
                    path1 = os.path.join(features_dir,u1,f1)
                    path2 = os.path.join(features_dir,u2,f2)
                    score = match_two_users(path1, path2)
                    if score > max_score:
                        max_score = score
            results[(u1,u2)] = max_score
    return results
