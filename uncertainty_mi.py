import math
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter

# 1. 准备一个小的 embedding 模型（速度快）
_EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_sentences(texts: List[str]) -> np.ndarray:
    """
    对一组句子做 embedding，返回 shape = (N, D) 的矩阵
    """
    return _EMB_MODEL.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算两向量的余弦相似度
    """
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return float(np.dot(a_norm, b_norm.T))


def cluster_texts(texts: List[str], sim_threshold: float = 0.85) -> List[int]:
    """
    简单的增量聚类：
    - 第一个样本开一个簇
    - 后面的样本与已有簇中心对比，如果相似度 >= 阈值就归入该簇，否则开新簇
    返回每个文本的 cluster_id 列表
    """
    if not texts:
        return []

    embeddings = embed_sentences(texts)

    cluster_centers = []  # list[np.ndarray]
    labels = []

    for i, emb in enumerate(embeddings):
        if len(cluster_centers) == 0:
            # 第一个样本，开新簇
            cluster_centers.append(emb)
            labels.append(0)
            continue

        # 找一个相似度 >= 阈值的簇
        assigned = False
        for cid, center in enumerate(cluster_centers):
            sim = cosine_similarity(emb, center)
            if sim >= sim_threshold:
                labels.append(cid)
                # 更新簇中心（简单平均）
                cluster_centers[cid] = (center + emb) / 2.0
                assigned = True
                break

        if not assigned:
            # 开新簇
            cluster_centers.append(emb)
            labels.append(len(cluster_centers) - 1)

    return labels


def estimate_mi_two_vars(
    a_labels: List[int],
    b_labels: List[int],
    gamma1: float = 1e-8,
    gamma2: float = 1e-8,
) -> float:
    """
    给定长度相同的两个整数标签序列 a_labels, b_labels（例如 cluster id），
    估计它们的互信息 I_hat。

    这里实现的是离散版 MI 估计器：
    I_hat = sum_{a,b} mu(a,b) * log((mu(a,b)+gamma1) / (mu1(a)*mu2(b)+gamma2))
    """
    assert len(a_labels) == len(b_labels), "a_labels 和 b_labels 长度必须一致"
    K = len(a_labels)
    if K == 0:
        return 0.0

    pairs = list(zip(a_labels, b_labels))
    joint_counts = Counter(pairs)
    a_counts = Counter(a_labels)
    b_counts = Counter(b_labels)

    joint_probs = {k: v / K for k, v in joint_counts.items()}
    a_probs = {k: v / K for k, v in a_counts.items()}
    b_probs = {k: v / K for k, v in b_counts.items()}

    I_hat = 0.0
    for (a, b), p_ab in joint_probs.items():
        p_a = a_probs[a]
        p_b = b_probs[b]
        p_prod = p_a * p_b

        num = p_ab + gamma1
        den = p_prod + gamma2
        I_hat += p_ab * math.log(num / (den + 1e-12) + 1e-12)

    return I_hat


def compute_uncertainty_from_mad_runs(
    A1_list: List[str],
    B1_list: List[str],
    sim_threshold: float = 0.85,
) -> Tuple[float, List[int], List[int]]:
    """
    给定同一个问题上，多次 MAD 辩论产生的：
    - A1_list: 每次正方第一轮回答
    - B1_list: 每次反方第一轮回答

    返回：
    - MI_hat: 互信息估计值（代表 epistemic uncertainty 程度的一个指标）
    - a_labels, b_labels: 聚类后的簇 id
    """
    assert len(A1_list) == len(B1_list), "A1_list 和 B1_list 长度必须一致"
    if len(A1_list) == 0:
        return 0.0, [], []

    # 1. 各自聚类
    a_labels = cluster_texts(A1_list, sim_threshold=sim_threshold)
    b_labels = cluster_texts(B1_list, sim_threshold=sim_threshold)

    # 2. 估计 MI
    mi_hat = estimate_mi_two_vars(a_labels, b_labels)

    return mi_hat, a_labels, b_labels


if __name__ == "__main__":
    # 一个简单示例：假设你已经对同一个问题跑了 5 次 MAD
    A1 = [
        "I think the correct translation is to eliminate an enemy division.",
        "The right translation should be to eliminate one division of the enemy.",
        "It means to wipe out a division of the enemy.",
        "It means to wipe out a division of the enemy.",
        "It means to totally eliminate an enemy division."
    ]

    B1 = [
        "No, it's more like to destroy one enemy division.",
        "I disagree, it should be rendered as destroying an enemy division.",
        "I also think the idea is to destroy one division of the enemy.",
        "I think the translation is to annihilate a division of the enemy.",
        "I disagree with the previous one, it's about destroying a division of the enemy."
    ]

    mi, a_ids, b_ids = compute_uncertainty_from_mad_runs(A1, B1)
    print("Estimated MI (epistemic uncertainty proxy):", mi)
    print("A1 clusters:", a_ids)
    print("B1 clusters:", b_ids)
