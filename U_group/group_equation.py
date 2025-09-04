import numpy as np
from typing import List, Optional, Sequence, Tuple

"""
Jensen-Shannon Divergence Implementations:

This module provides two approaches for computing Jensen-Shannon divergence:

1. EMBEDDING-SPACE APPROACH (Original):
   - jensen_shannon_stable(): Direct JSD over embeddings using sign-split pseudo-distributions
   - pairwise_u(): Understanding using embedding-space JSD
   - group_understanding_turn(): Group understanding with embedding-space metrics

2. HYPOTHESIS-SET APPROACH (Paper-faithful):
   - jensen_shannon_hypothesis_set(): JSD over shared hypothesis set H^(n)
   - jensen_shannon_from_restatements(): Complete pipeline with clustering
   - pairwise_u_hypothesis_based(): Understanding using hypothesis-set JSD
   - group_understanding_turn_hypothesis_based(): Group understanding with hypothesis-set metrics

The hypothesis-set approach follows the paper's definition more closely by:
- Clustering agent restatements to form shared hypothesis set H^(n)
- Computing belief distributions P_i^(n) over H^(n) using softmax
- Calculating JSD between these proper probability distributions
- Reducing embedding-space artifacts and providing cleaner information-fidelity signals

Usage example:
    # For hypothesis-set approach
    restatements = [emb1, emb2, emb3, ...]  # All agent restatement embeddings
    jsd = jensen_shannon_from_restatements(agent_i_emb, agent_j_emb, restatements)
    u_ij = pairwise_u_hypothesis_based(agent_i_emb, agent_j_emb, restatements)
"""

# -----------------------------
# Base utilities (as provided)
# -----------------------------

def jensen_shannon_stable(single_emb: np.ndarray, multi_emb: np.ndarray) -> float:
    """
    Stable Jensen-Shannon divergence using split positive/negative as pseudo-distributions.
    Returns a value in [0, 1] (clamped). Uses natural logs internally.
    """
    if single_emb.size == 0 or multi_emb.size == 0 or single_emb.shape != multi_emb.shape:
        raise ValueError("Embeddings must be non-empty and have matching shapes in jensen_shannon.")
    # Normalize embeddings to unit vectors
    epsilon = 1e-10
    single_norm = single_emb / (np.linalg.norm(single_emb) + epsilon)
    multi_norm = multi_emb / (np.linalg.norm(multi_emb) + epsilon)
    # Create pseudo-distributions by splitting positive and negative parts
    d = len(single_norm)
    p_s = np.zeros(2 * d)
    p_m = np.zeros(2 * d)
    for i in range(d):
        val_s = single_norm[i]
        if val_s >= 0:
            p_s[2 * i] = val_s
        else:
            p_s[2 * i + 1] = -val_s
        val_m = multi_norm[i]
        if val_m >= 0:
            p_m[2 * i] = val_m
        else:
            p_m[2 * i + 1] = -val_m
    # Normalize to proper probability distributions
    sum_s = np.sum(p_s) + epsilon
    sum_m = np.sum(p_m) + epsilon
    p_s /= sum_s
    p_m /= sum_m
    # Compute Jensen-Shannon divergence with smoothing
    m = 0.5 * (p_s + p_m)
    p_s_smooth = p_s + epsilon
    p_m_smooth = p_m + epsilon
    m_smooth = m + epsilon
    kl_sm = np.sum(p_s_smooth * np.log(p_s_smooth / m_smooth))
    kl_ms = np.sum(p_m_smooth * np.log(p_m_smooth / m_smooth))
    divergence = 0.5 * (kl_sm + kl_ms)
    return float(np.clip(divergence, 0.0, 1.0))

def jensen_shannon_hypothesis_set(
    emb_i: np.ndarray, 
    emb_j: np.ndarray, 
    hypothesis_embeddings: Sequence[np.ndarray],
    temperature: float = 1.0,
    eps: float = 1e-10
) -> float:
    """
    Jensen-Shannon divergence over shared hypothesis set H^(n) as defined in the paper.
    
    This implementation:
    1. Computes belief distributions P_i^(n) and P_j^(n) over the hypothesis set H^(n)
    2. Uses softmax with temperature to convert similarities to proper probability distributions  
    3. Computes standard Jensen-Shannon divergence between these belief distributions
    
    Args:
        emb_i: Agent i's understanding embedding
        emb_j: Agent j's understanding embedding  
        hypothesis_embeddings: Shared hypothesis set H^(n) for this turn
        temperature: Temperature parameter for softmax (default: 1.0)
        eps: Small epsilon for numerical stability (default: 1e-10)
        
    Returns:
        Jensen-Shannon divergence in [0, 1] between belief distributions
    """
    if len(hypothesis_embeddings) == 0:
        raise ValueError("hypothesis_embeddings must contain at least one hypothesis.")
    
    if emb_i.size == 0 or emb_j.size == 0:
        raise ValueError("Agent embeddings must be non-empty.")
        
    # Validate all embeddings have same dimensionality
    d = emb_i.shape[-1]
    if emb_j.shape[-1] != d:
        raise ValueError("Agent embeddings must have same dimensionality.")
    
    for h_emb in hypothesis_embeddings:
        if h_emb.size == 0 or h_emb.shape[-1] != d:
            raise ValueError("All hypothesis embeddings must be non-empty and have same dimensionality.")
    
    # Compute cosine similarities between each agent and each hypothesis
    similarities_i = []
    similarities_j = []
    
    for h_emb in hypothesis_embeddings:
        sim_i = cosine_sim(emb_i, h_emb)
        sim_j = cosine_sim(emb_j, h_emb)
        similarities_i.append(sim_i)
        similarities_j.append(sim_j)
    
    # Convert similarities to belief distributions using softmax
    similarities_i = np.array(similarities_i)
    similarities_j = np.array(similarities_j)
    
    # Apply temperature scaling and softmax
    logits_i = similarities_i / temperature
    logits_j = similarities_j / temperature
    
    # Numerical stability: subtract max before exponential
    logits_i = logits_i - np.max(logits_i)
    logits_j = logits_j - np.max(logits_j)
    
    exp_i = np.exp(logits_i)
    exp_j = np.exp(logits_j)
    
    # Belief distributions P_i^(n) and P_j^(n)
    P_i = exp_i / (np.sum(exp_i) + eps)
    P_j = exp_j / (np.sum(exp_j) + eps)
    
    # Compute Jensen-Shannon divergence between belief distributions
    # M = 0.5 * (P_i + P_j)
    M = 0.5 * (P_i + P_j)
    
    # Add smoothing for numerical stability
    P_i_smooth = P_i + eps
    P_j_smooth = P_j + eps  
    M_smooth = M + eps
    
    # KL divergences: KL(P_i || M) and KL(P_j || M)
    kl_i_m = np.sum(P_i_smooth * np.log(P_i_smooth / M_smooth))
    kl_j_m = np.sum(P_j_smooth * np.log(P_j_smooth / M_smooth))
    
    # Jensen-Shannon divergence
    js_div = 0.5 * (kl_i_m + kl_j_m)
    
    return float(np.clip(js_div, 0.0, 1.0))

def create_hypothesis_set_from_restatements(
    restatement_embeddings: Sequence[np.ndarray],
    similarity_threshold: float = 0.85,
    min_cluster_size: int = 2,
    max_hypotheses: int = 10
) -> List[np.ndarray]:
    """
    Create shared hypothesis set H^(n) by clustering agent restatements.
    
    This implements the paper's approach of forming hypothesis sets from clustered
    restatements rather than working directly in embedding space.
    
    Args:
        restatement_embeddings: All agent restatement embeddings for this turn
        similarity_threshold: Minimum cosine similarity for clustering (default: 0.85)
        min_cluster_size: Minimum cluster size to form a hypothesis (default: 2)  
        max_hypotheses: Maximum number of hypotheses to return (default: 10)
        
    Returns:
        List of hypothesis embeddings (cluster centroids)
    """
    if len(restatement_embeddings) == 0:
        return []
        
    if len(restatement_embeddings) == 1:
        return list(restatement_embeddings)
    
    # Convert to numpy array for easier manipulation
    embeddings = [np.array(emb) for emb in restatement_embeddings]
    n_restatements = len(embeddings)
    
    # Compute pairwise cosine similarities
    similarity_matrix = np.zeros((n_restatements, n_restatements))
    for i in range(n_restatements):
        for j in range(i, n_restatements):
            sim = cosine_sim(embeddings[i], embeddings[j])
            similarity_matrix[i, j] = similarity_matrix[j, i] = sim
    
    # Simple agglomerative clustering based on similarity threshold
    clusters = []
    used = set()
    
    for i in range(n_restatements):
        if i in used:
            continue
            
        # Start new cluster with embedding i
        cluster = [i]
        used.add(i)
        
        # Find all embeddings similar to i
        for j in range(i + 1, n_restatements):
            if j in used:
                continue
                
            # Check similarity to all members in current cluster
            similar_to_cluster = False
            for cluster_idx in cluster:
                if similarity_matrix[cluster_idx, j] >= similarity_threshold:
                    similar_to_cluster = True
                    break
            
            if similar_to_cluster:
                cluster.append(j)
                used.add(j)
        
        clusters.append(cluster)
    
    # Filter clusters by minimum size and create hypothesis embeddings
    hypotheses = []
    for cluster in clusters:
        if len(cluster) >= min_cluster_size:
            # Compute cluster centroid as hypothesis
            cluster_embeddings = [embeddings[idx] for idx in cluster]
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Normalize centroid
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
                
            hypotheses.append(centroid)
    
    # If no clusters meet minimum size, use individual embeddings as hypotheses
    if len(hypotheses) == 0:
        hypotheses = embeddings[:max_hypotheses]
    else:
        # Limit number of hypotheses
        hypotheses = hypotheses[:max_hypotheses]
    
    return hypotheses

def jensen_shannon_from_restatements(
    emb_i: np.ndarray,
    emb_j: np.ndarray, 
    all_restatement_embeddings: Sequence[np.ndarray],
    similarity_threshold: float = 0.85,
    min_cluster_size: int = 2,
    max_hypotheses: int = 10,
    temperature: float = 1.0,
    eps: float = 1e-10
) -> float:
    """
    Paper-faithful Jensen-Shannon divergence using clustered restatements as hypothesis set.
    
    This function implements the complete pipeline described in the paper:
    1. Clusters all agent restatements to form shared hypothesis set H^(n)
    2. Computes belief distributions P_i^(n) and P_j^(n) over H^(n) 
    3. Returns Jensen-Shannon divergence between these belief distributions
    
    Args:
        emb_i: Agent i's understanding embedding
        emb_j: Agent j's understanding embedding
        all_restatement_embeddings: All agent restatement embeddings for this turn
        similarity_threshold: Clustering similarity threshold (default: 0.85)
        min_cluster_size: Minimum cluster size to form hypothesis (default: 2)
        max_hypotheses: Maximum hypotheses in H^(n) (default: 10)
        temperature: Softmax temperature for belief distributions (default: 1.0)
        eps: Numerical stability parameter (default: 1e-10)
        
    Returns:
        Jensen-Shannon divergence in [0, 1] between agent belief distributions
    """
    # Create hypothesis set from clustered restatements
    hypothesis_set = create_hypothesis_set_from_restatements(
        all_restatement_embeddings,
        similarity_threshold=similarity_threshold,
        min_cluster_size=min_cluster_size,
        max_hypotheses=max_hypotheses
    )
    
    # If no hypothesis set could be created, fall back to direct embedding comparison
    if len(hypothesis_set) == 0:
        return jensen_shannon_stable(emb_i, emb_j)
    
    # Compute Jensen-Shannon divergence over hypothesis set
    return jensen_shannon_hypothesis_set(
        emb_i, emb_j, hypothesis_set, 
        temperature=temperature, eps=eps
    )

def cosine_sim(emb1: np.ndarray, emb2: np.ndarray) -> float:
    if emb1.size == 0 or emb2.size == 0 or emb1.shape != emb2.shape:
        raise ValueError("Embeddings must be non-empty and of the same shape in cosine_sim")
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 1.0
    cos = float(np.dot(emb1, emb2) / (norm1 * norm2))
    return float(np.clip(cos, -1.0, 1.0))

def semantic_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return float((1.0 - cosine_sim(emb1, emb2)) / 2.0)

def kappa(dist_A: float, dist_B: float) -> float:
    return float(np.exp(-(dist_A + dist_B) / 2.0))

def mutual_understanding(history_a: List[np.ndarray], history_b: List[np.ndarray], square: bool = False) -> float:
    """
    Two-agent discussion-level mutual understanding score (original).
    Linear weights: w_n = 2n / (T(T+1)).
    """
    if len(history_a) == 0 or len(history_b) == 0 or len(history_a) != len(history_b):
        raise ValueError("History lists must be of the same length and non-empty in mutual_understanding.")
    total = 0.0
    T = len(history_a)
    denom = T * (T + 1)
    for n in range(1, T + 1):
        i = n - 1
        if history_a[i].size == 0 or history_b[i].size == 0 or history_a[i].shape != history_b[i].shape:
            raise ValueError("History entries must be non-empty and matched in shape in mutual_understanding.")
        divergence = jensen_shannon_stable(history_a[i], history_b[i])
        align_prob = 1 - divergence
        d_sem = semantic_distance(history_a[i], history_b[i])
        align_sem = 1 - d_sem
        if n > 1:
            dist_prev_A = semantic_distance(history_a[i], history_a[i-1])
            dist_prev_B = semantic_distance(history_b[i], history_b[i-1])
            k = kappa(dist_prev_A, dist_prev_B)
        else:
            k = 1.0
        w_n = (2.0 * n) / denom
        total += w_n * k * (align_prob * align_sem)
    return float(total)

# ---------------------------------------------
# Group understanding utilities (new functions)
# ---------------------------------------------

def per_agent_kappa(current: np.ndarray, previous: Optional[np.ndarray], tau: float = 1.0) -> float:
    """
    κ_i^{(n)} = exp( - d_sem(current, previous) / τ ); if previous is None => 1.0.
    """
    if previous is None:
        return 1.0
    d = semantic_distance(current, previous)
    tau = max(float(tau), 1e-8)
    return float(np.exp(- d / tau))

def pairwise_u(emb_i: np.ndarray, emb_j: np.ndarray, eps: float = 1e-6) -> float:
    """
    u_{ij} = (1 - D_JS) * (1 - d_sem) ∈ [0,1]; floors at eps to avoid hard zeros in products.
    """
    djs = jensen_shannon_stable(emb_i, emb_j)
    align_prob = 1.0 - djs
    d_sem = semantic_distance(emb_i, emb_j)
    align_sem = 1.0 - d_sem
    u = float(align_prob * align_sem)
    if eps is not None and eps > 0.0:
        u = max(u, float(eps))
    return float(np.clip(u, 0.0, 1.0))

def pairwise_u_hypothesis_based(
    emb_i: np.ndarray, 
    emb_j: np.ndarray, 
    all_restatement_embeddings: Sequence[np.ndarray],
    similarity_threshold: float = 0.85,
    min_cluster_size: int = 2,
    max_hypotheses: int = 10,
    temperature: float = 1.0,
    eps: float = 1e-6
) -> float:
    """
    Paper-faithful pairwise understanding using hypothesis-set based Jensen-Shannon divergence.
    
    u_{ij} = (1 - D_JS_hypothesis) * (1 - d_sem) ∈ [0,1]
    
    Where D_JS_hypothesis is computed over clustered restatements forming hypothesis set H^(n),
    reducing embedding-space artifacts and providing cleaner information-fidelity signals.
    
    Args:
        emb_i: Agent i's understanding embedding
        emb_j: Agent j's understanding embedding  
        all_restatement_embeddings: All agent restatement embeddings for clustering
        similarity_threshold: Clustering similarity threshold (default: 0.85)
        min_cluster_size: Minimum cluster size to form hypothesis (default: 2)
        max_hypotheses: Maximum hypotheses in H^(n) (default: 10)
        temperature: Softmax temperature for belief distributions (default: 1.0)
        eps: Floor value to avoid hard zeros (default: 1e-6)
        
    Returns:
        Pairwise understanding score in [0, 1]
    """
    # Use hypothesis-based Jensen-Shannon divergence
    djs = jensen_shannon_from_restatements(
        emb_i, emb_j, all_restatement_embeddings,
        similarity_threshold=similarity_threshold,
        min_cluster_size=min_cluster_size,
        max_hypotheses=max_hypotheses,
        temperature=temperature,
        eps=eps
    )
    
    align_prob = 1.0 - djs
    d_sem = semantic_distance(emb_i, emb_j)
    align_sem = 1.0 - d_sem
    u = float(align_prob * align_sem)
    
    if eps is not None and eps > 0.0:
        u = max(u, float(eps))
    
    return float(np.clip(u, 0.0, 1.0))

def _validate_turn_embeddings(embs: Sequence[np.ndarray]) -> Tuple[int, int]:
    if len(embs) == 0:
        raise ValueError("At least one agent embedding is required for the turn.")
    d = int(embs[0].shape[-1])
    for e in embs:
        if e.size == 0 or e.shape[-1] != d:
            raise ValueError("All embeddings must be non-empty and have the same dimensionality.")
    return len(embs), d

def group_understanding_turn(
    embeddings: Sequence[np.ndarray],
    prev_embeddings: Optional[Sequence[Optional[np.ndarray]]] = None,
    tau: float = 1.0,
    eps: float = 1e-6,
    use_kappa: bool = True
) -> float:
    """
    U_group^{(n)} = (Π_{i<j} u_{ij})^{1/K} * (Π_i κ_i)^{1/N}
    """
    N, _ = _validate_turn_embeddings(embeddings)
    if prev_embeddings is not None and len(prev_embeddings) != N:
        raise ValueError("prev_embeddings must be None or have the same length as embeddings.")
    K = N * (N - 1) // 2
    if K == 0:
        k0 = per_agent_kappa(embeddings[0], None if prev_embeddings is None else prev_embeddings[0], tau=tau)
        return float(k0)
    prod_u = 1.0
    for i in range(N):
        for j in range(i + 1, N):
            prod_u *= pairwise_u(embeddings[i], embeddings[j], eps=eps)
    U_kappa = 1.0
    if use_kappa:
        prod_kappa = 1.0
        for i in range(N):
            prev_i = None if prev_embeddings is None else prev_embeddings[i]
            prod_kappa *= per_agent_kappa(embeddings[i], prev_i, tau=tau)
        U_kappa = prod_kappa ** (1.0 / N)
    U_pairs = prod_u ** (1.0 / K)
    return float(U_pairs * U_kappa)

def group_understanding_turn_gt(
    embeddings: Sequence[np.ndarray],
    gt_embedding: np.ndarray,
    prev_embeddings: Optional[Sequence[Optional[np.ndarray]]] = None,
    tau: float = 1.0,
    eps: float = 1e-6,
    use_kappa: bool = True
) -> float:
    """
    U_group,GT^{(n)} = (Π_i u_{i,GT} * Π_{i<j} u_{ij})^{1/(K+N)} * (Π_i κ_i)^{1/N}
    """
    N, _ = _validate_turn_embeddings(embeddings)
    if prev_embeddings is not None and len(prev_embeddings) != N:
        raise ValueError("prev_embeddings must be None or have the same length as embeddings.")
    if gt_embedding.size == 0 or gt_embedding.shape[-1] != embeddings[0].shape[-1]:
        raise ValueError("gt_embedding must be non-empty and match embedding dimensionality.")
    K = N * (N - 1) // 2
    prod_all = 1.0
    for i in range(N):
        prod_all *= pairwise_u(embeddings[i], gt_embedding, eps=eps)
    for i in range(N):
        for j in range(i + 1, N):
            prod_all *= pairwise_u(embeddings[i], embeddings[j], eps=eps)
    U_kappa = 1.0
    if use_kappa:
        prod_kappa = 1.0
        for i in range(N):
            prev_i = None if prev_embeddings is None else prev_embeddings[i]
            prod_kappa *= per_agent_kappa(embeddings[i], prev_i, tau=tau)
        U_kappa = prod_kappa ** (1.0 / N) if N > 0 else 1.0
    root = 1.0 / float(K + N) if (K + N) > 0 else 1.0
    U_pairs = prod_all ** root
    return float(U_pairs * U_kappa)

def group_understanding_turn_hypothesis_based(
    embeddings: Sequence[np.ndarray],
    all_restatement_embeddings: Sequence[np.ndarray],
    prev_embeddings: Optional[Sequence[Optional[np.ndarray]]] = None,
    tau: float = 1.0,
    eps: float = 1e-6,
    use_kappa: bool = True,
    similarity_threshold: float = 0.85,
    min_cluster_size: int = 2,
    max_hypotheses: int = 10,
    temperature: float = 1.0
) -> float:
    """
    Paper-faithful group understanding using hypothesis-set based Jensen-Shannon divergence.
    
    U_group^{(n)} = (Π_{i<j} u_{ij})^{1/K} * (Π_i κ_i)^{1/N}
    
    Where u_{ij} uses Jensen-Shannon divergence computed over clustered restatements 
    forming hypothesis set H^(n), providing cleaner information-fidelity signals.
    
    Args:
        embeddings: Current turn understanding embeddings for all agents
        all_restatement_embeddings: All agent restatement embeddings for clustering
        prev_embeddings: Previous turn embeddings (for kappa computation)
        tau: Temperature for kappa retention (default: 1.0)
        eps: Floor value to avoid hard zeros (default: 1e-6)
        use_kappa: Whether to include kappa factors (default: True)
        similarity_threshold: Clustering similarity threshold (default: 0.85)
        min_cluster_size: Minimum cluster size to form hypothesis (default: 2)
        max_hypotheses: Maximum hypotheses in H^(n) (default: 10)
        temperature: Softmax temperature for belief distributions (default: 1.0)
        
    Returns:
        Group understanding score using hypothesis-based approach
    """
    N, _ = _validate_turn_embeddings(embeddings)
    if prev_embeddings is not None and len(prev_embeddings) != N:
        raise ValueError("prev_embeddings must be None or have the same length as embeddings.")
    
    K = N * (N - 1) // 2
    if K == 0:
        k0 = per_agent_kappa(embeddings[0], None if prev_embeddings is None else prev_embeddings[0], tau=tau)
        return float(k0)
    
    # Use hypothesis-based pairwise understanding
    prod_u = 1.0
    for i in range(N):
        for j in range(i + 1, N):
            prod_u *= pairwise_u_hypothesis_based(
                embeddings[i], embeddings[j], all_restatement_embeddings,
                similarity_threshold=similarity_threshold,
                min_cluster_size=min_cluster_size,
                max_hypotheses=max_hypotheses,
                temperature=temperature,
                eps=eps
            )
    
    U_kappa = 1.0
    if use_kappa:
        prod_kappa = 1.0
        for i in range(N):
            prev_i = None if prev_embeddings is None else prev_embeddings[i]
            prod_kappa *= per_agent_kappa(embeddings[i], prev_i, tau=tau)
        U_kappa = prod_kappa ** (1.0 / N)
        
    U_pairs = prod_u ** (1.0 / K)
    return float(U_pairs * U_kappa)

def group_understanding_turn_hypothesis_based_gt(
    embeddings: Sequence[np.ndarray],
    gt_embedding: np.ndarray,
    all_restatement_embeddings: Sequence[np.ndarray],
    prev_embeddings: Optional[Sequence[Optional[np.ndarray]]] = None,
    tau: float = 1.0,
    eps: float = 1e-6,
    use_kappa: bool = True,
    similarity_threshold: float = 0.85,
    min_cluster_size: int = 2,
    max_hypotheses: int = 10,
    temperature: float = 1.0
) -> float:
    """
    Hypothesis-set group understanding for GT case.

    U_group,GT^{(n)} = (Π_i u_{i,GT} * Π_{i<j} u_{ij})^{1/(K+N)} * (Π_i κ_i)^{1/N}

    - u_{ij} uses hypothesis-based JSD over clustered restatements H^(n)
    - u_{i,GT} uses hypothesis-based JSD between agent i and GT over the same H^(n)
    """
    N, _ = _validate_turn_embeddings(embeddings)
    if prev_embeddings is not None and len(prev_embeddings) != N:
        raise ValueError("prev_embeddings must be None or have the same length as embeddings.")
    if gt_embedding.size == 0 or gt_embedding.shape[-1] != embeddings[0].shape[-1]:
        raise ValueError("gt_embedding must be non-empty and match embedding dimensionality.")

    K = N * (N - 1) // 2

    # Build hypothesis set from provided restatements
    # If caller passed raw embeddings, create_hypothesis_set_from_restatements will handle gracefully
    hypotheses = create_hypothesis_set_from_restatements(
        all_restatement_embeddings,
        similarity_threshold=similarity_threshold,
        min_cluster_size=min_cluster_size,
        max_hypotheses=max_hypotheses,
    )
    if len(hypotheses) == 0:
        # Fall back to traditional GT computation when no hypotheses
        return group_understanding_turn_gt(
            embeddings,
            gt_embedding,
            prev_embeddings=prev_embeddings,
            tau=tau,
            eps=eps,
            use_kappa=use_kappa,
        )

    # Product over agent↔GT and agent↔agent using hypothesis-based JSD
    prod_all = 1.0
    for i in range(N):
        # u_{i,GT}
        djs_i_gt = jensen_shannon_hypothesis_set(
            embeddings[i], gt_embedding, hypotheses, temperature=temperature, eps=eps
        )
        u_i_gt = max((1.0 - djs_i_gt) * (1.0 - semantic_distance(embeddings[i], gt_embedding)), float(eps))
        prod_all *= float(np.clip(u_i_gt, 0.0, 1.0))

    for i in range(N):
        for j in range(i + 1, N):
            djs_ij = jensen_shannon_hypothesis_set(
                embeddings[i], embeddings[j], hypotheses, temperature=temperature, eps=eps
            )
            u_ij = max((1.0 - djs_ij) * (1.0 - semantic_distance(embeddings[i], embeddings[j])), float(eps))
            prod_all *= float(np.clip(u_ij, 0.0, 1.0))

    U_kappa = 1.0
    if use_kappa:
        prod_kappa = 1.0
        for i in range(N):
            prev_i = None if prev_embeddings is None else prev_embeddings[i]
            prod_kappa *= per_agent_kappa(embeddings[i], prev_i, tau=tau)
        U_kappa = prod_kappa ** (1.0 / N) if N > 0 else 1.0

    root = 1.0 / float(K + N) if (K + N) > 0 else 1.0
    U_pairs = prod_all ** root
    return float(U_pairs * U_kappa)

# -----------------------------
# Discussion-level aggregators
# -----------------------------

def _linear_weights(T: int) -> List[float]:
    denom = T * (T + 1)
    return [ (2.0 * (n+1)) / denom for n in range(T) ]

def _uniform_weights(T: int) -> List[float]:
    return [1.0 / T for _ in range(T)]

def group_understanding_discussion(
    histories: Sequence[Sequence[np.ndarray]],
    weights: Optional[Sequence[float]] = None,
    tau: float = 1.0,
    eps: float = 1e-6,
    schedule: str = "linear",
    use_kappa: bool = True,
    use_hypothesis: bool = False,
    similarity_threshold: float = 0.85,
    min_cluster_size: int = 2,
    max_hypotheses: int = 10,
    temperature: float = 1.0,
) -> float:
    """
    histories: list (N agents) of lists (T turns) of embeddings.
    Returns sum_n w_n * U_group^{(n)} with default linear-in-time weights.
    """
    if len(histories) == 0:
        raise ValueError("histories must contain at least one agent.")
    N = len(histories)
    T = len(histories[0])
    if any(len(h) != T for h in histories):
        raise ValueError("All agents must have the same number of turns.")
    if weights is None:
        # For T total turns, we score turns 1 through T-1 (so T-1 scores need T-1 weights)
        weights = _linear_weights(T-1) if schedule == "linear" else _uniform_weights(T-1)
    else:
        if len(weights) != T-1:
            raise ValueError(f"weights must have length equal to the number of scored turns ({T-1}).")
        s = float(sum(weights))
        if s <= 0:
            raise ValueError("weights must sum to a positive value.")
        weights = [float(w) / s for w in weights]
    total = 0.0
    for n in range(1, T):
        curr = [histories[i][n] for i in range(N)]
        prev = None if n == 0 else [histories[i][n-1] for i in range(N)]
        if use_hypothesis:
            # Use current turn restatements (curr) as the hypothesis source
            u_n = group_understanding_turn_hypothesis_based(
                curr,
                all_restatement_embeddings=curr,
                prev_embeddings=prev,
                tau=tau,
                eps=eps,
                use_kappa=use_kappa,
                similarity_threshold=similarity_threshold,
                min_cluster_size=min_cluster_size,
                max_hypotheses=max_hypotheses,
                temperature=temperature,
            )
        else:
            u_n = group_understanding_turn(
                curr, prev_embeddings=prev, tau=tau, eps=eps, use_kappa=use_kappa
            )
        total += float(weights[n-1]) * u_n
    return float(total)

def group_understanding_discussion_gt(
    histories: Sequence[Sequence[np.ndarray]],
    gt_embedding: np.ndarray,
    weights: Optional[Sequence[float]] = None,
    tau: float = 1.0,
    eps: float = 1e-6,
    schedule: str = "linear",
    use_kappa: bool = True,
    use_hypothesis: bool = False,
    similarity_threshold: float = 0.85,
    min_cluster_size: int = 2,
    max_hypotheses: int = 10,
    temperature: float = 1.0,
) -> float:
    """
    histories: list (N agents) of lists (T turns) of embeddings.
    GT embedding is fixed across turns.
    """
    if len(histories) == 0:
        raise ValueError("histories must contain at least one agent.")
    N = len(histories)
    T = len(histories[0])
    if any(len(h) != T for h in histories):
        raise ValueError("All agents must have the same number of turns.")
    if weights is None:
        # For T total turns, we score turns 1 through T-1 (so T-1 scores need T-1 weights)
        weights = _linear_weights(T-1) if schedule == "linear" else _uniform_weights(T-1)
    else:
        if len(weights) != T-1:
            raise ValueError(f"weights must have length equal to the number of scored turns ({T-1}).")
        s = float(sum(weights))
        if s <= 0:
            raise ValueError("weights must sum to a positive value.")
        weights = [float(w) / s for w in weights]
    total = 0.0

    # range from 1 to T
    for n in range(1, T):
        curr = [histories[i][n] for i in range(N)]
        prev = None if n == 0 else [histories[i][n-1] for i in range(N)]
        if use_hypothesis:
            u_n = group_understanding_turn_hypothesis_based_gt(
                curr,
                gt_embedding=gt_embedding,
                all_restatement_embeddings=curr,
                prev_embeddings=prev,
                tau=tau,
                eps=eps,
                use_kappa=use_kappa,
                similarity_threshold=similarity_threshold,
                min_cluster_size=min_cluster_size,
                max_hypotheses=max_hypotheses,
                temperature=temperature,
            )
        else:
            u_n = group_understanding_turn_gt(
                curr,
                gt_embedding=gt_embedding,
                prev_embeddings=prev,
                tau=tau,
                eps=eps,
                use_kappa=use_kappa,
            )
        total += float(weights[n-1]) * u_n
    return float(total)

# ----------------------------------------
# Optional: diagnostics-friendly matrices
# ----------------------------------------

def pairwise_matrix_last_turn(histories: Sequence[Sequence[np.ndarray]], eps: float = 1e-6) -> np.ndarray:
    """
    Return the NxN matrix [u_{ij}^{(T)}] at the final turn T; diagonal = 1.
    """
    if len(histories) == 0:
        raise ValueError("histories must contain at least one agent.")
    N = len(histories)
    T = len(histories[0])
    if any(len(h) != T for h in histories):
        raise ValueError("All agents must have the same number of turns.")
    curr = [histories[i][-1] for i in range(N)]
    mat = np.eye(N, dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            u = pairwise_u(curr[i], curr[j], eps=eps)
            mat[i, j] = mat[j, i] = u
    return mat
