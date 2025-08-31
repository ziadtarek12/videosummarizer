"""
select.py
- compute multimodal scores
- method A: kmeans baseline
- method B: visual+audio fusion (weighted)
- method C: MMR + budget + optional actor quotas
- helper: frames_to_keyshots
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def method_A_kmeans(embs, k):
    k = min(k, len(embs))
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(embs)
    centers = km.cluster_centers_
    labels = km.labels_
    chosen = []
    for c in range(k):
        idxs = np.where(labels==c)[0]
        dists = np.linalg.norm(embs[idxs] - centers[c], axis=1)
        chosen.append(idxs[np.argmin(dists)])
    return sorted(chosen), labels

def method_B_visual_audio(embs, audio_scores, k, alpha=0.8):
    weights = 1.0 + alpha * audio_scores.reshape(-1,1)
    embs_w = embs * weights
    return method_A_kmeans(embs_w, k)

def method_C_mmr(embs, scores, k, lengths, budget_sec, lambda_param=0.7, quotas=None, actor_ids=None):
    sim = cosine_similarity(embs)
    relevance = scores.copy()
    selected = []
    total = 0.0
    candidates = list(range(len(embs)))
    while candidates and total < budget_sec and len(selected) < k:
        best = None; best_val = -1e9
        for i in candidates:
            if quotas is not None and actor_ids is not None:
                # simple quota check: actor quota is max fraction per actor
                actor = actor_ids[i]
                if actor is not None and actor!=-1:
                    # compute seconds already assigned to this actor in selected
                    assigned = sum(lengths[j] for j in selected if actor_ids[j]==actor)
                    if assigned + lengths[i] > quotas.get(actor, budget_sec):
                        continue
            if not selected:
                val = lambda_param * relevance[i]
            else:
                div = max(sim[i][j] for j in selected)
                val = lambda_param * relevance[i] - (1-lambda_param) * div
            if val > best_val:
                best_val = val; best = i
        if best is None:
            break
        if total + lengths[best] <= budget_sec:
            selected.append(best); total += lengths[best]
        candidates.remove(best)
    return sorted(selected)
    
def frames_to_keyshots(selected_idxs, timestamps, stride_sec, budget_sec):
    shots = []
    for i in selected_idxs:
        s = max(0, timestamps[i] - stride_sec)
        e = timestamps[i] + stride_sec
        shots.append([s,e])
    shots = sorted(shots, key=lambda x: x[0])
    kept = []
    total = 0.0
    for s,e in shots:
        d = e - s
        if total + d <= budget_sec:
            kept.append([s,e]); total += d
    return kept
