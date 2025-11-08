import json
import os
import math
from typing import List, Dict, Tuple
from itertools import combinations

Minutia = Dict[str, float]

# ========================
# Funzioni base
# ========================

def load_minutiae(file_path: str) -> List[Minutia]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Manteniamo anche minutiae di qualità leggermente bassa per Bozorth3
    return [m for m in data if m['quality'] >= 0.2]

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def angle_between(p1: Tuple[float,float], p2: Tuple[float,float]) -> float:
    return math.atan2(p2[1]-p1[1], p2[0]-p1[0])

def save_results(results: Dict[Tuple[str,str], float], output_file: str):
    """
    Salva i risultati del batch matching in un file JSON.
    La chiave (user1, user2) viene convertita in stringa 'user1_vs_user2'.
    """
    json_dict = {f"{u1}_vs_{u2}": score for (u1, u2), score in results.items()}
    with open(output_file, 'w') as f:
        json.dump(json_dict, f, indent=4)
    print(f"Risultati salvati in {output_file}")

# ========================
# Creazione coppie di minutiae
# ========================

def build_pairs(minutiae: List[Minutia]) -> List[Tuple[Tuple[int,int], float, float]]:
    """
    Ritorna lista di coppie di minutiae:
    ((idx1, idx2), distanza, angolo)
    """
    pairs = []
    for (i1, m1), (i2, m2) in combinations(enumerate(minutiae), 2):
        if m1['type'] != m2['type']:
            continue
        d = euclidean_distance((m1['x'], m1['y']), (m2['x'], m2['y']))
        a = angle_between((m1['x'], m1['y']), (m2['x'], m2['y']))
        pairs.append(((i1, i2), d, a))
    return pairs

# ========================
# Matching coppie
# ========================

def match_pairs(pairs1, pairs2, dist_thresh=25.0, angle_thresh=0.5) -> int:
    """
    Confronta due insiemi di coppie di minutiae.
    Ritorna il numero di coppie corrispondenti.
    """
    matched = 0
    for (_, d1, a1) in pairs1:
        for (_, d2, a2) in pairs2:
            if abs(d1 - d2) <= dist_thresh and abs(a1 - a2) <= angle_thresh:
                matched += 1
                break
    return matched

# ========================
# Matching tra due utenti
# ========================

def match_two_users(user1_path: str, user2_path: str) -> float:
    minutiae1 = load_minutiae(user1_path)
    minutiae2 = load_minutiae(user2_path)
    if len(minutiae1) < 2 or len(minutiae2) < 2:
        return 0.0
    pairs1 = build_pairs(minutiae1)
    pairs2 = build_pairs(minutiae2)
    matched_pairs = match_pairs(pairs1, pairs2)
    total_pairs = max(len(pairs1), len(pairs2))
    return matched_pairs / total_pairs if total_pairs > 0 else 0.0

# ========================
# Batch matching
# ========================

def batch_match(features_dir: str) -> Dict[Tuple[str,str], float]:
    results = {}
    results_file = os.path.join(features_dir, "batch_match_results.json")
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
        
        save_results(results, results_file)

    return results
