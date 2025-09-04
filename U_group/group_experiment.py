import os
import sys
import time
import json
import argparse
from typing import List, Sequence, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
import openai

# --- metrics / equation layer ---
from group_equation import (
    pairwise_u,
    pairwise_u_hypothesis_based,
    group_understanding_turn,
    group_understanding_turn_gt,  # not used in Case A, but available
    group_understanding_turn_hypothesis_based,
    group_understanding_turn_hypothesis_based_gt,
    group_understanding_discussion,
    group_understanding_discussion_gt
)

# ----------------------------
# Configuration (edit freely)
# ----------------------------

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# If you run a local compatible endpoint, set LOCAL_AI_API_URL and pass model="local"
LOCAL_AI_API_URL = os.getenv("LOCAL_AI_API_URL", "http://localhost:8080/v1")

# Models (you can change any of these)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
AGENTS_CONFIG = os.getenv("AGENTS", "gpt-5-nano,gpt-5-nano,gpt-5-nano,gpt-5-nano")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "local")   # placeholder for local endpoints

# Experiment controls
MAX_TURNS = 8             # upper cap (actual number may be smaller if early stop)
MIN_TURNS = 4             # minimum turns to run
TAU = 1.0                 # κ_i retention temperature
EPSILON = 1e-6            # product floor (prevents hard zeros)
SLEEP_BETWEEN_CALLS = 0.5     # API pacing (seconds)

# Early stopping criteria
CONVERGENCE_VARIANCE_THRESHOLD = 0.0001    # variance threshold for convergence detection
CONVERGENCE_IMPROVEMENT_THRESHOLD = 0.001  # improvement threshold for convergence detection

VERBOSE = False

CASE_A = "Agent Generated Q/GT"
CASE_B = "External Q/GT"

# ----------------------------
# API helpers
# ----------------------------

def embed(text: str) -> np.ndarray:
    """Embed text; returns 1D np.ndarray (empty on failure)."""
    try:
        resp = openai.embeddings.create(model=EMBEDDING_MODEL, input=text)
        v = np.array(resp.data[0].embedding, dtype=float)
        return v
    except Exception as e:
        print(f"[embed] Error: {e}")
        return np.array([])

def _chat_openai(model: str, messages: List[dict], temperature: float = 0.7, max_tokens: int = 300) -> str:
    try:
        resp = openai.chat.completions.create(
            model=model,
            messages=messages,
            # temperature=temperature,
            # max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[chat] API error: {e}")
        return ""

def _chat_local(messages: List[dict], base_url: str = LOCAL_AI_API_URL, temperature: float = 0.7, max_tokens: int = 300) -> str:
    try:
        client = openai.Client(base_url=base_url, api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=LOCAL_MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[chat-local] Error: {e}")
        return ""

def chat(messages: List[dict], model: str, temperature: float = 0.7, max_tokens: int = 300, use_local: bool = False) -> str:
    if use_local:
        return _chat_local(messages, temperature=temperature, max_tokens=max_tokens)
    return _chat_openai(model, messages, temperature=temperature, max_tokens=max_tokens)

# ----------------------------
# Helper functions for Case A
# ----------------------------

def generate_initial_query(topic_seed: str, speaker_index: int, agent_models: Sequence[str], use_local: bool = False) -> str:
    """Generate the initial seed query from the topic seed."""
    query_prompt = [
        {
            "role": "system",
            "content": (
                "Generate a specific query that that is complex enough to require additional clarification.\n"
                "Make it concrete enough to be answerable but ambiguous enough to require discussion."
            )
        },
        {"role": "user", "content": f"Topic seed: {topic_seed}\n\nGenerate the technical query:"},
    ]
    seed_query = chat(query_prompt, model=agent_models[speaker_index], temperature=0.6, max_tokens=100, use_local=use_local)
    if not seed_query:
        seed_query = f"How should we evaluate {topic_seed} with specific metrics and constraints?"
    return seed_query

def generate_speaker_understanding(seed_query: str, speaker_index: int, agent_models: Sequence[str], use_local: bool = False) -> str:
    """Generate the speaker's initial understanding of their own query."""
    understanding_prompt = [
        {
            "role": "system",
            "content": "You created this query. State your intended understanding in 1-2 clear sentences."
        },
        {"role": "user", "content": f"Your query: {seed_query}\n\nWhat is your understanding of this query?"}
    ]
    return chat(understanding_prompt, model=agent_models[speaker_index], temperature=0.6, max_tokens=150, use_local=use_local)

def generate_understanding_statement(
    agent_idx: int,
    agent_names: Sequence[str],
    agent_models: Sequence[str],
    seed_query: str,
    qa_conversation: List[str],
    previous_understanding: str,
    temperature: float = 0.7,
    use_local: bool = False,
    is_initial: bool = False
) -> str:
    """Generate an understanding statement for an agent."""
    if is_initial:
        # Initial understanding based on query only
        prompt = [
            {
                "role": "system",
                "content": f"You are {agent_names[agent_idx]}. State your understanding of the given query in 1-2 clear sentences."
            },
            {"role": "user", "content": f"Query: {seed_query}\n\nWhat is your understanding of this query?"}
        ]
        fallback = f"I understand this as a question about {seed_query.split()[0] if seed_query else 'the topic'}."
    else:
        # Updated understanding based on Q&A history
        prompt = [
            {
                "role": "system",
                "content": (
                    f"You are {agent_names[agent_idx]}. Update your understanding of the original query "
                    "based on the Q&A discussion. State your refined understanding in 1-2 clear sentences."
                )
            },
            {
                "role": "user",
                "content": (
                    f"ORIGINAL QUERY: {seed_query}\n\n"
                    f"Q&A DISCUSSION:\n{chr(10).join(qa_conversation)}\n\n"
                    f"Your previous understanding: {previous_understanding}\n\n"
                    "What is your updated understanding of the original query?"
                )
            }
        ]
        fallback = previous_understanding

    text = chat(prompt, model=agent_models[agent_idx], temperature=temperature, max_tokens=150, use_local=use_local)
    return text if text else fallback

def generate_clarifying_question(
    questioner_idx: int,
    questioner_name: str,
    agent_models: Sequence[str],
    seed_query: str,
    qa_conversation: List[str],
    temperature: float = 0.7,
    use_local: bool = False
) -> str:
    """Generate a clarifying question from a questioner agent."""
    question_prompt = [
        {
            "role": "system",
            "content": (
                f"You are {questioner_name}. Ask a clarifying question about the original query.\n"
                "Focus on technical details, ambiguous terms, or missing specifications.\n"
                "Ask ONE specific question that would help clarify the query."
            )
        },
        {
            "role": "user",
            "content": (
                f"ORIGINAL QUERY: {seed_query}\n\n"
                f"Q&A HISTORY:\n{chr(10).join(qa_conversation)}\n\n"
                "Ask a clarifying question that would help you better understand the original query."
            )
        }
    ]

    question = chat(question_prompt, model=agent_models[questioner_idx], temperature=temperature, max_tokens=100, use_local=use_local)
    return question if question else "Could you clarify the specific requirements and constraints?"

def generate_answer(
    speaker_index: int,
    speaker_name: str,
    agent_models: Sequence[str],
    seed_query: str,
    qa_conversation: List[str],
    question: str,
    temperature: float = 0.7,
    use_local: bool = False
) -> str:
    """Generate an answer from the speaker to a clarifying question."""
    answer_prompt = [
        {
            "role": "system",
            "content": (
                f"You are {speaker_name}, the query owner. Answer the clarifying question directly and specifically.\n"
                "Provide technical details, constraints, or specifications that address the question."
            )
        },
        {
            "role": "user",
            "content": (
                f"YOUR ORIGINAL QUERY: {seed_query}\n\n"
                f"Q&A HISTORY:\n{chr(10).join(qa_conversation)}\n\n"
                f"QUESTION: {question}\n\n"
                "Provide a clear, specific answer:"
            )
        }
    ]

    answer = chat(answer_prompt, model=agent_models[speaker_index], temperature=temperature, max_tokens=200, use_local=use_local)
    return answer if answer else "Let me clarify the technical requirements and constraints."

def safe_embed_with_fallback(text: str, previous_embedding: Optional[np.ndarray] = None) -> np.ndarray:
    """Safely embed text with fallback to previous embedding or random vector."""
    v = embed(text)
    if v.size == 0:
        if previous_embedding is not None:
            return previous_embedding
        return np.random.default_rng().normal(size=768)
    return v

def update_all_agent_understanding(
    N: int,
    agent_names: Sequence[str],
    agent_models: Sequence[str],
    seed_query: str,
    qa_conversation: List[str],
    understanding_statements: List[str],
    embeddings_hist: List[List[np.ndarray]],
    temperature: float = 0.7,
    use_local: bool = False
) -> List[str]:
    """Update understanding statements for all agents and their embeddings."""
    new_understanding: List[str] = []
    for i in range(N):
        text = generate_understanding_statement(
            i, agent_names, agent_models, seed_query, qa_conversation, understanding_statements[i], temperature, use_local
        )
        new_understanding.append(text)
        if VERBOSE:
            print(f"  {agent_names[i]} ({agent_models[i]}): {text}")

        # Embed understanding statement with fallback to previous embedding
        v = safe_embed_with_fallback(text, embeddings_hist[i][-1])
        embeddings_hist[i].append(v)
        time.sleep(SLEEP_BETWEEN_CALLS)

    return new_understanding

def check_convergence(
    turn_scores: List[float],
    discussion_scores: List[float],
    n: int,
    min_turns: int,
    convergence_variance_threshold: float,
    convergence_improvement_threshold: float
) -> Tuple[bool, str]:
    """
    Check if convergence criteria are met.

    Returns:
        Tuple of (should_stop, convergence_type)
    """
    should_stop = False
    convergence_type = ""

    if n >= min_turns and len(turn_scores) >= 3:
        # Check both turn scores and discussion scores for convergence
        last_3_turns = turn_scores[-3:]
        last_3_discussion = discussion_scores[-3:]

        # Calculate variance and mean improvement for sliding window
        turn_variance = np.var(last_3_turns)
        discussion_variance = np.var(last_3_discussion)

        # Calculate average improvement over the window
        turn_improvements = [last_3_turns[i+1] - last_3_turns[i] for i in range(2)]
        discussion_improvements = [last_3_discussion[i+1] - last_3_discussion[i] for i in range(2)]

        avg_turn_improvement = np.mean(turn_improvements)
        avg_discussion_improvement = np.mean(discussion_improvements)

        # Convergence criteria: low variance AND small average improvement
        turn_converged = turn_variance < convergence_variance_threshold and abs(avg_turn_improvement) < convergence_improvement_threshold
        discussion_converged = discussion_variance < convergence_variance_threshold and abs(avg_discussion_improvement) < convergence_improvement_threshold

        if VERBOSE:
            print(f"[DEBUG] Sliding window analysis:")
            print(f"  Turn scores variance: {turn_variance:.6f}, avg improvement: {avg_turn_improvement:.6f}")
            print(f"  Discussion scores variance: {discussion_variance:.6f}, avg improvement: {avg_discussion_improvement:.6f}")
            print(f"  Turn converged: {turn_converged}, Discussion converged: {discussion_converged}")

        # Stop if either metric shows convergence
        if turn_converged or discussion_converged:
            should_stop = True
            convergence_type = "turn scores" if turn_converged else "discussion scores"

    return should_stop, convergence_type

# ----------------------------
# Experiment runner (Case A)
# ----------------------------

def run_case_A(
    topic_seed: str,
    agent_names: Sequence[str],
    agent_models: Sequence[str],
    speaker_index: int = 0,
    max_turns: int = MAX_TURNS,
    min_turns: int = MIN_TURNS,
    tau: float = TAU,
    eps: float = EPSILON,
    use_local: bool = False,
    temperature: float = 0.7,
    use_kappa: bool = True,
    schedule: str = "linear",
    convergence_variance_threshold: float = CONVERGENCE_VARIANCE_THRESHOLD,
    convergence_improvement_threshold: float = CONVERGENCE_IMPROVEMENT_THRESHOLD,
    hyp_djs: bool = False,
    hyp_temperature: float = 1.0,
) -> Dict:
    """
    New Case A:
    1. Speaker creates query and their understanding (only query shared)
    2. All agents generate initial understanding statements (private)
    3. Other agents ask clarifying questions in order, speaker answers
    4. After each Q&A, all agents update understanding statements (private)
    5. Only understanding statements are used for U_group calculation
    """
    assert len(agent_names) == len(agent_models), "agent_names and agent_models must align"
    N = len(agent_names)
    assert 1 <= N, "Need at least one agent"

    # 1) Speaker generates the seed query AND their initial understanding
    speaker_name = agent_names[speaker_index]

    # Generate query and speaker understanding
    seed_query = generate_initial_query(topic_seed, speaker_index, agent_models, use_local)
    speaker_understanding = generate_speaker_understanding(seed_query, speaker_index, agent_models, use_local)

    if VERBOSE:
        print(f"[DEBUG] Speaker {speaker_name} ({agent_models[speaker_index]}) generated query: {seed_query}")
        print(f"[DEBUG] Speaker understanding: {speaker_understanding}")

    time.sleep(SLEEP_BETWEEN_CALLS)

    # 2) Initialize conversation histories (separate for Q&A and understanding)
    qa_conversation = [f"[{speaker_name}] {seed_query}"]  # Q&A conversation history
    understanding_statements: List[str] = [""] * N  # Current understanding for each agent
    embeddings_hist: List[List[np.ndarray]] = [[] for _ in range(N)]  # Understanding embeddings only

    # Track understanding evolution over time
    understanding_evolution: List[List[str]] = [[] for _ in range(N)]  # understanding_evolution[agent_i][turn_j]

    # All agents generate initial understanding statements (including speaker)
    if VERBOSE:
        print(f"[DEBUG] Turn 1 - All agents generate initial understanding:")

    for i in range(N):
        if i == speaker_index:
            # Speaker uses their pre-generated understanding
            understanding_statements[i] = speaker_understanding
        else:
            # Other agents generate understanding based on query only
            understanding_statements[i] = generate_understanding_statement(
                i, agent_names, agent_models, seed_query, [], "", temperature, use_local, is_initial=True
            )
        if VERBOSE:
            print(f"  {agent_names[i]} ({agent_models[i]}): {understanding_statements[i]}")

        # Track understanding evolution (Turn 1)
        understanding_evolution[i].append(understanding_statements[i])

        # Embed understanding statement with fallback
        v = safe_embed_with_fallback(understanding_statements[i])
        embeddings_hist[i].append(v)
        time.sleep(SLEEP_BETWEEN_CALLS)

    # 3) Q&A Phase: Other agents ask questions, speaker answers
    turn_scores: List[float] = []
    discussion_scores: List[float] = []
    questioners: List[str] = []
    questioner_models: List[str] = []  # Track the models of questioners
    questions: List[str] = []
    answers: List[str] = []

    # Skip scoring turn 1 (initial understanding generation)
    if VERBOSE:
        print(f"[DEBUG] Turn 1 - Initial understanding generation (no scoring)")

    T = max_turns
    # Track subcomponents per turn
    C_series: List[float] = []
    T_series: List[float] = []
    K_series: List[float] = []
    for n in range(2, max_turns + 1):
        # Select questioner (cycle through non-speaker agents)
        non_speaker_indices = [i for i in range(N) if i != speaker_index]
        if not non_speaker_indices:
            break  # Only one agent, can't ask questions

        questioner_idx = non_speaker_indices[(n - 2) % len(non_speaker_indices)]
        questioner_name = agent_names[questioner_idx]
        questioner_model = agent_models[questioner_idx]

        if VERBOSE:
            print(f"\n[DEBUG] Turn {n} - {questioner_name} ({questioner_model}) asks clarifying question:")

        # Questioner asks clarifying question
        question = generate_clarifying_question(
            questioner_idx, questioner_name, agent_models, seed_query, qa_conversation, temperature, use_local
        )

        qa_conversation.append(f"[{questioner_name} Question]: {question}")
        questions.append(question)
        questioners.append(questioner_name)
        questioner_models.append(questioner_model)  # Track questioner model

        if VERBOSE:
            print(f"  Question: {question}")

        time.sleep(SLEEP_BETWEEN_CALLS)

        # Speaker answers the question
        answer = generate_answer(
            speaker_index, speaker_name, agent_models, seed_query, qa_conversation, question, temperature, use_local
        )

        qa_conversation.append(f"[{speaker_name} Answer]: {answer}")
        answers.append(answer)

        if VERBOSE:
            print(f"  Answer: {answer}")

        time.sleep(SLEEP_BETWEEN_CALLS)

        # 4) All agents update their understanding based on Q&A history
        if VERBOSE:
            print(f"[DEBUG] Turn {n} - All agents update understanding after Q&A:")

        understanding_statements = update_all_agent_understanding(
            N, agent_names, agent_models, seed_query, qa_conversation,
            understanding_statements, embeddings_hist, temperature, use_local
        )

        # Track understanding evolution for this turn
        for i in range(N):
            understanding_evolution[i].append(understanding_statements[i])

        # 5) Compute understanding scores based on understanding statements only
        curr = [embeddings_hist[i][-1] for i in range(N)]
        prev = [embeddings_hist[i][-2] for i in range(N)]

        if hyp_djs:
            # Use hypothesis-based approach with all current restatement embeddings
            all_restatements = curr  # All current turn embeddings serve as restatements
            u_combined, C_turn, K_turn = group_understanding_turn_hypothesis_based(
                curr, all_restatements, prev_embeddings=prev,
                tau=tau, eps=eps, use_kappa=use_kappa, temperature=hyp_temperature
            )
            T_turn = np.nan  # Not applicable for non-GT
        else:
            # Use traditional embedding-space approach
            u_combined, C_turn, K_turn = group_understanding_turn(
                curr, prev_embeddings=prev, tau=tau, eps=eps, use_kappa=use_kappa
            )
            T_turn = np.nan  # Not applicable for non-GT

        # Record per-turn aggregates
        turn_scores.append(u_combined)
        C_series.append(C_turn)
        T_series.append(T_turn)
        K_series.append(K_turn)

        if VERBOSE:
            print(f"[DEBUG] Turn {n} pairwise understanding similarities:")

        # Compute cumulative discussion-level score
        # For turn n (where n >= 2), we have embeddings from turn 1 to turn n
        # We want to include all turns up to current turn n
        histories_subset = [embeddings_hist[i][:n] for i in range(N)]  # Include turns 1 to n
        U_discussion_cumulative = group_understanding_discussion(
            histories=histories_subset,
            weights=None,
            tau=tau,
            eps=eps,
            schedule=schedule,
            use_kappa=use_kappa,
            use_hypothesis=hyp_djs,
            temperature=hyp_temperature
        )
        discussion_scores.append(U_discussion_cumulative)

        print(
            f"Turn {n} metrics: U_turn = {u_combined:.4f}, u_ij = {C_turn:.4f}, "
            f"kappa = {K_turn:.4f}, U_discussion_cumulative = {U_discussion_cumulative:.4f}"
        )

        # Early-stop: use sliding window on last 3 values to detect convergence
        should_stop, convergence_type = check_convergence(
            turn_scores, discussion_scores, n, min_turns,
            convergence_variance_threshold, convergence_improvement_threshold
        )

        if should_stop:
            T = n
            if VERBOSE:
                print(f"[DEBUG] Early stopping at turn {T} due to {convergence_type} convergence")
            break

    # Final discussion-level aggregate using group_understanding_discussion()
    # Use embeddings from understanding statements only
    actual_turns = T - 1  # Number of understanding turns we scored (starting from turn 2)
    histories = [embeddings_hist[i][:actual_turns+1] for i in range(N)]  # Include turn 1 + scored turns
    U_discussion_final = group_understanding_discussion(
        histories=histories,
        weights=None,  # Use default linear weights
        tau=tau,
        eps=eps,
        schedule=schedule,
    use_kappa=use_kappa,
    use_hypothesis=hyp_djs,
    temperature=hyp_temperature
    )

    print(f"Final U_discussion = {U_discussion_final}")

    # Final pairwise matrix at last turn (based on understanding statements)
    final_embs = [embeddings_hist[i][-1] for i in range(N)]
    pairwise_mat = np.eye(N, dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            if hyp_djs:
                u_ij = pairwise_u_hypothesis_based(final_embs[i], final_embs[j], final_embs, eps=eps, temperature=hyp_temperature)
            else:
                u_ij = pairwise_u(final_embs[i], final_embs[j], eps=eps)
            pairwise_mat[i, j] = pairwise_mat[j, i] = u_ij

    # Generate final weights for DataFrame
    final_weights = [(2.0 * n) / (actual_turns * (actual_turns + 1)) for n in range(1, actual_turns + 1)]

    # Build a tidy per-turn DataFrame (Q&A turns with understanding scores)
    df = pd.DataFrame({
        "turn": list(range(2, T + 1)),  # Turns 2 to T
        "U_group_turn": turn_scores,  # Understanding scores
        "U_discussion_cumulative": discussion_scores,  # Cumulative scores
        "C_turn": C_series,
        "T_turn": T_series,
        "K_turn": K_series,
        "questioner": questioners[:actual_turns],  # Who asked the question
        "questioner_model": questioner_models[:actual_turns],  # Model used by questioner
        "question": questions[:actual_turns],  # The clarifying questions
        "answer": answers[:actual_turns],  # Speaker's answers
        "weight": final_weights
    })

    return {
        "seed_query": seed_query,
        "qa_conversation": qa_conversation,  # Q&A conversation history
        "understanding_statements": understanding_statements,  # Final understanding statements
        "understanding_evolution": understanding_evolution,  # Understanding evolution per agent per turn
        "per_turn_df": df,
        "U_discussion": U_discussion_final,
        "turn_scores": turn_scores,
        "discussion_scores": discussion_scores,
        "pairwise_matrix": pairwise_mat,
        "agent_names": agent_names,
        "agent_models": agent_models,  # Include the actual models being used
        "speaker_model": agent_models[speaker_index],  # Model used by the speaker
        "embeddings_hist": embeddings_hist,  # Understanding embeddings only
        "questions": questions,
        "answers": answers,
        "questioners": questioners,
    "questioner_models": questioner_models,  # Models used by questioners
    "hyp_djs": hyp_djs,
    "hyp_temperature": hyp_temperature,
    }

# ----------------------------
# Helper functions for Case GT
# ----------------------------

def generate_gt_clarifying_question(
    questioner_idx: int,
    questioner_name: str,
    agent_models: Sequence[str],
    seed_query: str,
    gt_statement: str,
    qa_conversation: List[str],
    temperature: float = 0.7,
    use_local: bool = False
) -> str:
    """Generate a clarifying question from a questioner agent based on query and GT statement."""
    question_prompt = [
        {
            "role": "system",
            "content": (
                f"You are {questioner_name}. Ask a clarifying question about the query.\n"
                "Focus on technical details, ambiguous terms, or missing specifications.\n"
                "Ask ONE specific question that would help clarify the query."
            )
        },
        {
            "role": "user",
            "content": (
                f"QUERY: {seed_query}\n\n"
                f"REFERENCE UNDERSTANDING: {gt_statement}\n\n"
                f"Q&A HISTORY:\n{chr(10).join(qa_conversation)}\n\n"
                "Ask a clarifying question that would help you better understand the query."
            )
        }
    ]

    question = chat(question_prompt, model=agent_models[questioner_idx], temperature=temperature, max_tokens=100, use_local=use_local)
    return question if question else "Could you clarify the specific requirements and constraints?"

def generate_gt_answer(
    answerer_idx: int,
    answerer_name: str,
    agent_models: Sequence[str],
    seed_query: str,
    gt_statement: str,
    qa_conversation: List[str],
    question: str,
    temperature: float = 0.7,
    use_local: bool = False
) -> str:
    """Generate an answer from any agent to a clarifying question."""
    answer_prompt = [
        {
            "role": "system",
            "content": (
                f"You are {answerer_name}. Answer the clarifying question based on your understanding of the query.\n"
                "Provide technical details, constraints, or specifications that address the question."
            )
        },
        {
            "role": "user",
            "content": (
                f"QUERY: {seed_query}\n\n"
                f"REFERENCE UNDERSTANDING: {gt_statement}\n\n"
                f"Q&A HISTORY:\n{chr(10).join(qa_conversation)}\n\n"
                f"QUESTION: {question}\n\n"
                "Provide a clear, specific answer:"
            )
        }
    ]

    answer = chat(answer_prompt, model=agent_models[answerer_idx], temperature=temperature, max_tokens=200, use_local=use_local)
    return answer if answer else "Let me clarify the technical requirements and constraints."

def generate_gt_understanding_statement(
    agent_idx: int,
    agent_names: Sequence[str],
    agent_models: Sequence[str],
    seed_query: str,
    gt_statement: str,
    qa_conversation: List[str],
    previous_understanding: str,
    temperature: float = 0.7,
    use_local: bool = False,
    is_initial: bool = False
) -> str:
    """Generate an understanding statement for an agent in GT scenario."""
    if is_initial:
        # Initial understanding based on query only
        prompt = [
            {
                "role": "system",
                "content": f"You are {agent_names[agent_idx]}. State your understanding of the given query in 1-2 clear sentences."
            },
            {"role": "user", "content": f"Query: {seed_query}\n\nWhat is your understanding of this query?"}
        ]
        fallback = f"I understand this as a question about {seed_query.split()[0] if seed_query else 'the topic'}."
    else:
        # Updated understanding based on Q&A history
        prompt = [
            {
                "role": "system",
                "content": (
                    f"You are {agent_names[agent_idx]}. Update your understanding of the original query "
                    "based on the Q&A discussion. State your refined understanding in 1-2 clear sentences."
                )
            },
            {
                "role": "user",
                "content": (
                    f"ORIGINAL QUERY: {seed_query}\n\n"
                    f"Q&A DISCUSSION:\n{chr(10).join(qa_conversation)}\n\n"
                    f"Your previous understanding: {previous_understanding}\n\n"
                    "What is your updated understanding of the original query?"
                )
            }
        ]
        fallback = previous_understanding

    text = chat(prompt, model=agent_models[agent_idx], temperature=temperature, max_tokens=150, use_local=use_local)
    return text if text else fallback

# ----------------------------
# Experiment runner (Case GT)
# ----------------------------

def run_case_GT(
    seed_query: str,
    gt_statement: str,
    agent_names: Sequence[str],
    agent_models: Sequence[str],
    max_turns: int = MAX_TURNS,
    min_turns: int = MIN_TURNS,
    tau: float = TAU,
    eps: float = EPSILON,
    use_local: bool = False,
    temperature: float = 0.7,
    use_kappa: bool = True,
    schedule: str = "linear",
    convergence_variance_threshold: float = CONVERGENCE_VARIANCE_THRESHOLD,
    convergence_improvement_threshold: float = CONVERGENCE_IMPROVEMENT_THRESHOLD,
    hyp_djs: bool = False,
    hyp_temperature: float = 1.0,
) -> Dict:
    """
    Case GT (Ground Truth):
    1. Query and GT understanding statement are provided as parameters
    2. All agents generate initial understanding statements (private)
    3. Random agents ask clarifying questions and random agents answer them
    4. After each Q&A, all agents update understanding statements (private)
    5. Only understanding statements are used for U_group calculation, compared against static GT embedding
    """
    assert len(agent_names) == len(agent_models), "agent_names and agent_models must align"
    N = len(agent_names)
    assert N >= 2, "Need at least two agents for GT case"

    if VERBOSE:
        print(f"[DEBUG] Running Case GT with query: {seed_query}")
        print(f"[DEBUG] Ground truth statement: {gt_statement}")

    # Embed the GT statement once (it remains static)
    gt_embedding = safe_embed_with_fallback(gt_statement)

    time.sleep(SLEEP_BETWEEN_CALLS)

    # Initialize conversation histories
    qa_conversation = [f"[Query] {seed_query}"]  # Q&A conversation history
    understanding_statements: List[str] = [""] * N  # Current understanding for each agent
    embeddings_hist: List[List[np.ndarray]] = [[] for _ in range(N)]  # Understanding embeddings only

    # Track understanding evolution over time
    understanding_evolution: List[List[str]] = [[] for _ in range(N)]

    # All agents generate initial understanding statements
    if VERBOSE:
        print(f"[DEBUG] Turn 1 - All agents generate initial understanding:")

    for i in range(N):
        understanding_statements[i] = generate_gt_understanding_statement(
            i, agent_names, agent_models, seed_query, gt_statement, [], "", temperature, use_local, is_initial=True
        )
        if VERBOSE:
            print(f"  {agent_names[i]} ({agent_models[i]}): {understanding_statements[i]}")

        # Track understanding evolution (Turn 1)
        understanding_evolution[i].append(understanding_statements[i])

        # Embed understanding statement with fallback
        v = safe_embed_with_fallback(understanding_statements[i])
        embeddings_hist[i].append(v)
        time.sleep(SLEEP_BETWEEN_CALLS)

    # Q&A Phase: Random agents ask questions and random agents answer
    turn_scores: List[float] = []
    discussion_scores: List[float] = []
    questioners: List[str] = []
    questioner_models: List[str] = []
    answerers: List[str] = []
    answerer_models: List[str] = []
    questions: List[str] = []
    answers: List[str] = []

    # Skip scoring turn 1 (initial understanding generation)
    if VERBOSE:
        print(f"[DEBUG] Turn 1 - Initial understanding generation (no scoring)")

    # Use random seed based on current time for reproducible but varied results
    rng = np.random.default_rng(int(time.time()) % 2**32)

    T = max_turns
    # Track subcomponents per turn (C: prod_gt, T: prod_ij, K: kappa)
    C_series: List[float] = []
    T_series: List[float] = []
    K_series: List[float] = []
    for n in range(2, max_turns + 1):
        # Select random questioner and answerer (must be different)
        available_indices = list(range(N))
        questioner_idx = rng.choice(available_indices)
        available_for_answer = [i for i in available_indices if i != questioner_idx]
        answerer_idx = rng.choice(available_for_answer)

        questioner_name = agent_names[questioner_idx]
        questioner_model = agent_models[questioner_idx]
        answerer_name = agent_names[answerer_idx]
        answerer_model = agent_models[answerer_idx]

        if VERBOSE:
            print(f"\n[DEBUG] Turn {n} - {questioner_name} ({questioner_model}) asks question, {answerer_name} ({answerer_model}) answers:")

        # Questioner asks clarifying question
        question = generate_gt_clarifying_question(
            questioner_idx, questioner_name, agent_models, seed_query, gt_statement, qa_conversation, temperature, use_local
        )

        qa_conversation.append(f"[{questioner_name} Question]: {question}")
        questions.append(question)
        questioners.append(questioner_name)
        questioner_models.append(questioner_model)
        answerers.append(answerer_name)
        answerer_models.append(answerer_model)

        if VERBOSE:
            print(f"  Question: {question}")

        time.sleep(SLEEP_BETWEEN_CALLS)

        # Answerer responds to the question
        answer = generate_gt_answer(
            answerer_idx, answerer_name, agent_models, seed_query, gt_statement, qa_conversation, question, temperature, use_local
        )

        qa_conversation.append(f"[{answerer_name} Answer]: {answer}")
        answers.append(answer)

        if VERBOSE:
            print(f"  Answer: {answer}")

        time.sleep(SLEEP_BETWEEN_CALLS)

        # All agents update their understanding based on Q&A history
        if VERBOSE:
            print(f"[DEBUG] Turn {n} - All agents update understanding after Q&A:")

        for i in range(N):
            text = generate_gt_understanding_statement(
                i, agent_names, agent_models, seed_query, gt_statement, qa_conversation, understanding_statements[i], temperature, use_local
            )
            understanding_statements[i] = text
            if VERBOSE:
                print(f"  {agent_names[i]} ({agent_models[i]}): {text}")

            # Track understanding evolution for this turn
            understanding_evolution[i].append(text)

            # Embed understanding statement with fallback to previous embedding
            v = safe_embed_with_fallback(text, embeddings_hist[i][-1])
            embeddings_hist[i].append(v)
            time.sleep(SLEEP_BETWEEN_CALLS)

        # Compute understanding scores using GT functions
        curr = [embeddings_hist[i][-1] for i in range(N)]
        prev = [embeddings_hist[i][-2] for i in range(N)]

        # Use hypothesis-based approach when enabled; otherwise traditional
        if hyp_djs:
            # IMPORTANT: include GT in the hypothesis set for GT-case consistency
            curr_with_gt = list(curr) + [gt_embedding]
            u_combined, T_turn, C_turn, K_turn = group_understanding_turn_hypothesis_based_gt(
                curr,
                gt_embedding,
                all_restatement_embeddings=curr_with_gt,
                prev_embeddings=prev,
                tau=tau,
                eps=eps,
                use_kappa=use_kappa,
                temperature=hyp_temperature,
            )
        else:
            u_combined, T_turn, C_turn, K_turn = group_understanding_turn_gt(
                curr, gt_embedding, prev_embeddings=prev, tau=tau, eps=eps, use_kappa=use_kappa
            )
        turn_scores.append(u_combined)
        # Record subcomponents
        C_series.append(C_turn)
        T_series.append(T_turn)
        K_series.append(K_turn)

        if VERBOSE:
            print(f"[DEBUG] Turn {n} GT-based understanding similarities:")

        # Compute cumulative discussion-level score using GT
        histories_subset = [embeddings_hist[i][:n] for i in range(N)]  # Include turns 1 to n
        U_discussion_cumulative = group_understanding_discussion_gt(
            histories=histories_subset,
            gt_embedding=gt_embedding,
            weights=None,
            tau=tau,
            eps=eps,
            schedule=schedule,
            use_kappa=use_kappa,
            use_hypothesis=hyp_djs,
            temperature=hyp_temperature,
        )
        discussion_scores.append(U_discussion_cumulative)

        print(
            f"Turn {n} metrics: U_turn_GT = {u_combined:.4f}, GT = {T_turn:.4f}, "
            f"u_ij = {C_turn:.4f}, kappa = {K_turn:.4f}, "
            f"U_discussion_cumulative_GT = {U_discussion_cumulative:.4f}"
        )

        # Early-stop: use sliding window on last 3 values to detect convergence
        should_stop, convergence_type = check_convergence(
            turn_scores, discussion_scores, n, min_turns,
            convergence_variance_threshold, convergence_improvement_threshold
        )

        if should_stop:
            T = n
            if VERBOSE:
                print(f"[DEBUG] Early stopping at turn {T} due to {convergence_type} convergence")
            break

    # Final discussion-level aggregate using GT function
    actual_turns = T - 1  # Number of understanding turns we scored (starting from turn 2)
    histories = [embeddings_hist[i][:actual_turns+1] for i in range(N)]  # Include turn 1 + scored turns
    U_discussion_final = group_understanding_discussion_gt(
        histories=histories,
        gt_embedding=gt_embedding,
        weights=None,  # Use default linear weights
        tau=tau,
        eps=eps,
        schedule=schedule,
        use_kappa=use_kappa,
        use_hypothesis=hyp_djs,
        temperature=hyp_temperature
    )

    print(f"Final U_discussion_GT = {U_discussion_final}")

    # Final GT-based similarity matrix at last turn (consistent with hyp_djs)
    final_embs = [embeddings_hist[i][-1] for i in range(N)]
    gt_similarity_matrix = np.zeros((N, 1), dtype=float)  # N agents x 1 GT
    for i in range(N):
        if hyp_djs:
            # Include GT in hypothesis set for agent↔GT measurement
            restatements = list(final_embs) + [gt_embedding]
            u_i_gt = pairwise_u_hypothesis_based(final_embs[i], gt_embedding, restatements, eps=eps, temperature=hyp_temperature)
        else:
            u_i_gt = pairwise_u(final_embs[i], gt_embedding, eps=eps)
        gt_similarity_matrix[i, 0] = u_i_gt

    # Create pairwise matrix including GT as an additional "agent"
    # This allows us to use the same plotting functions as Case A
    extended_embs = final_embs + [gt_embedding]  # Add GT embedding as last "agent"
    extended_N = N + 1
    pairwise_matrix = np.eye(extended_N, dtype=float)

    # Fill in pairwise similarities (agents + GT)
    for i in range(extended_N):
        for j in range(i + 1, extended_N):
            if hyp_djs:
                # Build restatements including GT to share the same H^(n) across all pairs
                restatements = list(final_embs) + [gt_embedding]
                u_ij = pairwise_u_hypothesis_based(extended_embs[i], extended_embs[j], restatements, eps=eps, temperature=hyp_temperature)
            else:
                u_ij = pairwise_u(extended_embs[i], extended_embs[j], eps=eps)
            pairwise_matrix[i, j] = pairwise_matrix[j, i] = u_ij

    # GT "agent" has perfect self-understanding (kappa = 1.0)
    pairwise_matrix[-1, -1] = 1.0

    # Extended agent names including GT
    extended_agent_names = list(agent_names) + ["GT"]
    extended_agent_models = list(agent_models) + ["ground_truth"]

    # Generate final weights for DataFrame
    final_weights = [(2.0 * n) / (actual_turns * (actual_turns + 1)) for n in range(1, actual_turns + 1)]

    # Build a tidy per-turn DataFrame
    df = pd.DataFrame({
        "turn": list(range(2, T + 1)),  # Turns 2 to T
        "U_group_turn_gt": turn_scores,  # GT-based understanding scores
        "U_discussion_cumulative_gt": discussion_scores,  # GT-based cumulative scores
        "C_turn": C_series,
        "T_turn": T_series,
        "K_turn": K_series,
        "questioner": questioners[:actual_turns],  # Who asked the question
        "questioner_model": questioner_models[:actual_turns],  # Model used by questioner
        "answerer": answerers[:actual_turns],  # Who answered the question
        "answerer_model": answerer_models[:actual_turns],  # Model used by answerer
        "question": questions[:actual_turns],  # The clarifying questions
        "answer": answers[:actual_turns],  # The answers
        "weight": final_weights
    })

    return {
        "seed_query": seed_query,
        "gt_statement": gt_statement,
        "qa_conversation": qa_conversation,  # Q&A conversation history
        "understanding_statements": understanding_statements,  # Final understanding statements
        "understanding_evolution": understanding_evolution,  # Understanding evolution per agent per turn
        "per_turn_df": df,
        "U_discussion": U_discussion_final,
        "turn_scores": turn_scores,
        "discussion_scores": discussion_scores,
        "gt_similarity_matrix": gt_similarity_matrix,  # Final GT similarities (N x 1)
        "pairwise_matrix": pairwise_matrix,  # Extended pairwise matrix including GT as agent
        "gt_embedding": gt_embedding,  # Include GT embedding for reference
        "agent_names": extended_agent_names,  # Include GT as "agent" for plotting
        "agent_models": extended_agent_models,  # Include GT model for plotting
        "original_agent_names": agent_names,  # Original agent names (without GT)
        "original_agent_models": agent_models,  # Original agent models (without GT)
        "embeddings_hist": embeddings_hist,  # Understanding embeddings only (original agents)
        "questions": questions,
        "answers": answers,
        "questioners": questioners,
        "questioner_models": questioner_models,
        "answerers": answerers,
        "answerer_models": answerer_models,
        "hyp_djs": hyp_djs,
        "hyp_temperature": hyp_temperature,
    }

# ----------------------------
# Plotting helpers (matplotlib)
# ----------------------------
def plot_per_turn_scores(df: pd.DataFrame, title: str = "Group Understanding Metrics"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Determine which columns to use based on what's available
    if "U_group_turn_gt" in df.columns:  # Case GT
        turn_col = "U_group_turn_gt"
        cumulative_col = "U_discussion_cumulative_gt"
        turn_label = "U_group^(n)_GT (per-turn)"
        cumulative_label = "U_discussion_GT (cumulative)"
    else:  # Case A
        turn_col = "U_group_turn"
        cumulative_col = "U_discussion_cumulative"
        turn_label = "U_group^(n) (per-turn)"
        cumulative_label = "U_discussion (cumulative)"

    # Plot per-turn scores
    ax1.plot(df["turn"], df[turn_col], marker="o", label=turn_label, color="blue")
    ax1.set_xlabel("Turn")
    ax1.set_ylabel("U_group^(n)")
    ax1.set_title("Per-turn Group Understanding")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    # Plot cumulative discussion scores
    ax2.plot(df["turn"], df[cumulative_col], marker="s", label=cumulative_label, color="red")
    ax2.set_xlabel("Turn")
    ax2.set_ylabel("U_discussion")
    ax2.set_title("Cumulative Discussion Understanding")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    plt.tight_layout()

def plot_pairwise_evolution(embeddings_hist: List[List[np.ndarray]], agent_names: Sequence[str],
                           start_turn: int = 2, eps: float = 1e-6, use_hypothesis: bool = False,
                           hyp_temperature: float = 1.0) -> None:
    """
    Plot pairwise understanding matrices for each turn starting from start_turn.

    Args:
        embeddings_hist: List of embedding histories for each agent
        agent_names: Names of the agents
        start_turn: Turn to start plotting from (1-indexed, default=2)
        eps: Epsilon value for pairwise_u calculation
    """
    N = len(agent_names)
    max_turns = len(embeddings_hist[0])

    # Calculate number of plots needed
    num_turns = max_turns - start_turn + 1
    if num_turns <= 0:
        print("No turns to plot")
        return

    # Create subplot grid
    cols = min(3, num_turns)  # Max 3 columns
    rows = (num_turns + cols - 1) // cols  # Ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if num_turns == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()

    for turn_idx, turn in enumerate(range(start_turn, max_turns + 1)):
        # Calculate pairwise matrix for this turn
        turn_embs = [embeddings_hist[i][turn-1] for i in range(N)]
        pairwise_mat = np.eye(N, dtype=float)
        for i in range(N):
            for j in range(i + 1, N):
                if use_hypothesis:
                    # Build restatements from all entities present at this turn (agents and possibly GT)
                    restatements = turn_embs
                    u = pairwise_u_hypothesis_based(embeddings_hist[i][turn-1], embeddings_hist[j][turn-1], restatements, eps=eps, temperature=hyp_temperature)
                else:
                    u = pairwise_u(embeddings_hist[i][turn-1], embeddings_hist[j][turn-1], eps=eps)
                pairwise_mat[i, j] = pairwise_mat[j, i] = u

        # Plot heatmap
        ax = axes[turn_idx]
        im = ax.imshow(pairwise_mat, interpolation="nearest", vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(range(len(agent_names)))
        ax.set_yticks(range(len(agent_names)))
        ax.set_xticklabels(agent_names, rotation=45, ha="right")
        ax.set_yticklabels(agent_names)
        ax.set_title(f"Turn {turn}")

        # Add colorbar to each subplot
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add text annotations for values
        for i in range(N):
            for j in range(N):
                text = ax.text(j, i, f'{pairwise_mat[i, j]:.3f}',
                             ha="center", va="center", color="white" if pairwise_mat[i, j] < 0.5 else "black",
                             fontsize=8)

    # Hide unused subplots
    for idx in range(num_turns, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Pairwise Understanding Evolution", fontsize=16)
    plt.tight_layout()

def plot_combined_metrics(df: pd.DataFrame, title: str = "Group Understanding Metrics"):
    plt.figure(figsize=(10, 6))

    # Determine which columns to use based on what's available
    if "U_group_turn_gt" in df.columns:  # Case GT
        turn_col = "U_group_turn_gt"
        cumulative_col = "U_discussion_cumulative_gt"
        turn_label = "U_group^(n)_GT (per-turn)"
        cumulative_label = "U_discussion_GT (cumulative)"
    else:  # Case A
        turn_col = "U_group_turn"
        cumulative_col = "U_discussion_cumulative"
        turn_label = "U_group^(n) (per-turn)"
        cumulative_label = "U_discussion (cumulative)"

    # Helper to find first existing column name from options
    def _pick_col(options):
        for c in options:
            if c in df.columns:
                return c
        return None

    # Base plots: per-turn U and cumulative U
    plt.plot(df["turn"], df[turn_col], marker="o", label=turn_label, linewidth=2, color="#1f77b4")
    plt.plot(df["turn"], df[cumulative_col], marker="s", label=cumulative_label, linewidth=2, color="#d62728")

    # Optional: plot C, T, K if available
    c_col = _pick_col(["C_turn", "C"])  # unified across cases
    t_col = _pick_col(["T_turn", "T"])  # unified across cases
    k_col = _pick_col(["K_turn", "K"])  # unified across cases

    # Use distinct styles/colors for clarity
    if c_col is not None:
        plt.plot(df["turn"], df[c_col], marker="^", linestyle="-.", linewidth=1.8, label="C (consensus)", color="#2ca02c")
    else:
        # Soft hint for users running interactively
        print("[plot_combined_metrics] C-series not found in DataFrame (expected 'C_turn' or 'C').")

    if t_col is not None:
        plt.plot(df["turn"], df[t_col], marker="v", linestyle=":", linewidth=1.8, label="T (truth/GT)", color="#9467bd")
    else:
        print("[plot_combined_metrics] T-series not found in DataFrame (expected 'T_turn' or 'T').")

    if k_col is not None:
        plt.plot(df["turn"], df[k_col], marker="D", linestyle="--", linewidth=1.8, label="K (kappa)", color="#8c564b")
    else:
        print("[plot_combined_metrics] K-series not found in DataFrame (expected 'K_turn' or 'K').")

    plt.xlabel("Turn")
    plt.ylabel("Understanding Score")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

def plot_pairwise_heatmap(mat: np.ndarray, labels: Sequence[str], title: str = "Final pairwise u_ij"):
    plt.figure(figsize=(5.5, 5))
    plt.imshow(mat, interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.tight_layout()

def save_plots_to_folder(results: Dict, output_dir: str) -> None:
    """
    Save experiment plots as PNG files to the specified directory.

    Args:
        results: Dictionary containing experiment results from run_case_A
        output_dir: Directory to save plots in
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save per-turn scores plot (dual subplot version)
    plot_per_turn_scores(results["per_turn_df"])
    per_turn_path = os.path.join(output_dir, "understanding_metrics.png")
    plt.savefig(per_turn_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Understanding metrics plot saved to: {per_turn_path}")

    # Save combined metrics plot
    plot_combined_metrics(results["per_turn_df"])
    combined_path = os.path.join(output_dir, "combined_metrics.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined metrics plot saved to: {combined_path}")

    # Save pairwise plots (now available for both Case A and Case GT)
    # Save pairwise heatmap (final turn)
    plot_pairwise_heatmap(results["pairwise_matrix"], results["agent_names"])
    heatmap_path = os.path.join(output_dir, "pairwise_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pairwise heatmap saved to: {heatmap_path}")

    # Save pairwise evolution plot (all turns starting from turn 2)
    # For Case GT, we need to create extended embeddings history including GT
    if "gt_embedding" in results:  # Case GT
        # Create extended embeddings history with GT as constant final agent
        extended_embeddings_hist = []
        for i in range(len(results["original_agent_names"])):
            extended_embeddings_hist.append(results["embeddings_hist"][i])
        # Add GT embedding history (constant across all turns)
        gt_history = [results["gt_embedding"]] * len(results["embeddings_hist"][0])
        extended_embeddings_hist.append(gt_history)

        plot_pairwise_evolution(
            extended_embeddings_hist,
            results["agent_names"],
            start_turn=2,
            eps=1e-6,
            use_hypothesis=results.get("hyp_djs", False),
            hyp_temperature=results.get("hyp_temperature", 1.0)
        )
    else:  # Case A
        plot_pairwise_evolution(
            results["embeddings_hist"],
            results["agent_names"],
            start_turn=2,
            eps=1e-6,
            use_hypothesis=results.get("hyp_djs", False),
            hyp_temperature=results.get("hyp_temperature", 1.0)
        )

    evolution_path = os.path.join(output_dir, "pairwise_evolution.png")
    plt.savefig(evolution_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pairwise evolution plot saved to: {evolution_path}")


def create_experiment_folder(base_name: str = "experiment") -> str:
    """
    Create a unique folder for experiment results with timestamp.

    Args:
        base_name: Base name for the folder

    Returns:
        Path to the created folder
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    folder_name = f"{base_name}_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

def save_experiment_complete(results: Dict, local_llm: bool, output_dir: str = None, include_embeddings: bool = False,
                           save_plots: bool = True, show_plots: bool = False) -> str:
    """
    Save complete experiment results including JSON data and plots to a unique folder.

    Args:
        results: Dictionary containing experiment results from run_case_A
        output_dir: Output directory (if None, creates timestamped folder)
        include_embeddings: Whether to include embeddings in JSON output
        save_plots: Whether to save plots as PNG files
        show_plots: Whether to display plots interactively

    Returns:
        Path to the output directory
    """
    # Create output directory if not provided
    if output_dir is None:
        output_dir = create_experiment_folder("experiment")
    else:
        os.makedirs(output_dir, exist_ok=True)

    # Save JSON results
    json_path = os.path.join(output_dir, "results.json")
    save_experiment_results(results, local_llm, json_path, include_embeddings=include_embeddings)

    # Save plots
    if save_plots:
        save_plots_to_folder(results, output_dir)

    # Optionally show plots
    if show_plots:
        plot_per_turn_scores(results["per_turn_df"])
        plot_combined_metrics(results["per_turn_df"])
        # Both cases now have pairwise_matrix
        plot_pairwise_heatmap(results["pairwise_matrix"], results["agent_names"])

        # Handle different embeddings history for evolution plot
        if "gt_embedding" in results:  # Case GT
            # Create extended embeddings history with GT as constant final agent
            extended_embeddings_hist = []
            for i in range(len(results["original_agent_names"])):
                extended_embeddings_hist.append(results["embeddings_hist"][i])
            # Add GT embedding history (constant across all turns)
            gt_history = [results["gt_embedding"]] * len(results["embeddings_hist"][0])
            extended_embeddings_hist.append(gt_history)
            plot_pairwise_evolution(extended_embeddings_hist, results["agent_names"], start_turn=2, eps=1e-6)
        else:  # Case A
            plot_pairwise_evolution(results["embeddings_hist"], results["agent_names"], start_turn=2, eps=1e-6)

        plt.show()

    return output_dir

# ----------------------------
# Results persistence
# ----------------------------

def save_experiment_results(results: Dict, local_llm: bool, filename: str, include_embeddings: bool = False) -> None:
    """
    Save experiment results to a JSON file with optimized structure.

    This version eliminates data duplication and includes intermediate understanding evolution.
    Handles both Case A (agent-to-agent) and Case GT (agent-to-ground-truth) results.

    Args:
        results: Dictionary containing experiment results from run_case_A or run_case_GT
        filename: Output filename (will add .json if not present)
        include_embeddings: Whether to include embeddings in the output (default: False due to size)
    """
    if not filename.endswith('.json'):
        filename += '.json'

    # Detect experiment type
    is_gt_case = "gt_statement" in results

    # Create optimized serializable structure
    serializable_results = {
        # Experiment setup
        "experiment_setup": {
            "experiment_type": CASE_B if is_gt_case else CASE_A,
            "local_llm": local_llm,
            "seed_query": results["seed_query"],
            "agent_names": list(results["agent_names"]),
            "agent_models": list(results["agent_models"]),
            "total_turns": len(results["per_turn_df"]),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },

        # Final results summary
        "final_results": {
            "U_discussion_final": float(results["U_discussion"]),
            "final_understanding_statements": results["understanding_statements"]
        },

        # Complete Q&A conversation history (for reference)
        "full_conversation": results["qa_conversation"],

        # Understanding evolution over time (NEW: eliminates missing intermediate states)
        "understanding_evolution": {
            agent_name: [
                {
                    "turn": turn_idx + 1,  # Turn numbers start at 1
                    "understanding": understanding_text
                }
                for turn_idx, understanding_text in enumerate(agent_understandings)
            ]
            for agent_name, agent_understandings in zip(
                results["agent_names"],
                results["understanding_evolution"]
            )
        },

        # Metrics time series
        "metrics_evolution": {
            "turn_scores": [float(x) for x in results["turn_scores"]],
            "discussion_scores": [float(x) for x in results["discussion_scores"]]
        }
    }

    # Add experiment-specific data
    if is_gt_case:
        # Case GT specific fields
        serializable_results["experiment_setup"]["gt_statement"] = results["gt_statement"]
        serializable_results["final_results"]["gt_similarity_matrix"] = results["gt_similarity_matrix"].tolist()

        # Turn-by-turn conversation and evolution for GT case
        serializable_results["conversation_evolution"] = [
            {
                "turn": int(results["per_turn_df"]["turn"][i]),
                "questioner": results["per_turn_df"]["questioner"][i],
                "questioner_model": results["per_turn_df"]["questioner_model"][i],
                "answerer": results["per_turn_df"]["answerer"][i],
                "answerer_model": results["per_turn_df"]["answerer_model"][i],
                "question": results["per_turn_df"]["question"][i],
                "answer": results["per_turn_df"]["answer"][i],
                "metrics": {
                    "U_group_turn_gt": float(results["per_turn_df"]["U_group_turn_gt"][i]),
                    "U_discussion_cumulative_gt": float(results["per_turn_df"]["U_discussion_cumulative_gt"][i]),
                    "weight": float(results["per_turn_df"]["weight"][i])
                }
            }
            for i in range(len(results["per_turn_df"]["turn"]))
        ]
    else:
        # Case A specific fields
        serializable_results["experiment_setup"]["speaker_model"] = results["speaker_model"]
        serializable_results["final_results"]["pairwise_understanding_matrix"] = results["pairwise_matrix"].tolist()

        # Turn-by-turn conversation and evolution for Case A
        serializable_results["conversation_evolution"] = [
            {
                "turn": int(results["per_turn_df"]["turn"][i]),
                "questioner": results["per_turn_df"]["questioner"][i],
                "questioner_model": results["per_turn_df"]["questioner_model"][i],
                "question": results["per_turn_df"]["question"][i],
                "answer": results["per_turn_df"]["answer"][i],
                "metrics": {
                    "U_group_turn": float(results["per_turn_df"]["U_group_turn"][i]),
                    "U_discussion_cumulative": float(results["per_turn_df"]["U_discussion_cumulative"][i]),
                    "weight": float(results["per_turn_df"]["weight"][i])
                }
            }
            for i in range(len(results["per_turn_df"]["turn"]))
        ]

    # Optionally include embeddings (can be large)
    if include_embeddings:
        serializable_results["embeddings_history"] = [
            [emb.tolist() for emb in agent_embeddings]
            for agent_embeddings in results["embeddings_hist"]
        ]
        if is_gt_case:
            serializable_results["gt_embedding"] = results["gt_embedding"].tolist()

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"Optimized experiment results saved to: {filename}")
        print(f"Structure: {serializable_results['experiment_setup']['experiment_type']}, "
              f"{len(serializable_results['conversation_evolution'])} turns, "
              f"{len(serializable_results['experiment_setup']['agent_names'])} agents, "
              f"{'with' if include_embeddings else 'without'} embeddings")
    except Exception as e:
        print(f"Error saving results to {filename}: {e}")

# ----------------------------
# Command line argument parsing
# ----------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for experiment parameters."""
    parser = argparse.ArgumentParser(
        description="Run multi-agent communication experiment with group understanding metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    global VERBOSE

    # Experiment parameters
    parser.add_argument("-m", "--max_turns", type=int, default=MAX_TURNS,
                       help="Maximum number of turns in the experiment")
    parser.add_argument("--min_turns", type=int, default=MIN_TURNS,
                       help="Minimum number of turns before early stopping")
    parser.add_argument("--tau", type=float, default=TAU,
                       help="Kappa retention temperature parameter")
    parser.add_argument("-e", "--epsilon",  type=float, default=EPSILON,
                       help="Product floor to prevent hard zeros")
    parser.add_argument("-t", "--temperature", type=float, default=0.7,
                       help="LLM sampling temperature")
    parser.add_argument("-l", "--use_local", action="store_true",
                       help="Use local AI endpoint instead of OpenAI")
    parser.add_argument("-k", "--use_kappa", action="store_true",
                       help="Enable kappa-based metrics (if implemented)")
    parser.add_argument("--schedule", type=str, default="linear",
                       choices=["linear", "uniform"],
                       help="Weighting schedule for discussion-level understanding (linear or uniform)")
    parser.add_argument("--convergence_variance_threshold", type=float, default=CONVERGENCE_VARIANCE_THRESHOLD,
                       help="Variance threshold for early stopping convergence detection")
    parser.add_argument("--convergence_improvement_threshold", type=float, default=CONVERGENCE_IMPROVEMENT_THRESHOLD,
                       help="Average improvement threshold for early stopping convergence detection")

    # Additional parameters
    parser.add_argument("--topic_seed", type=str,
                       default="Evaluating group consensus in multi-agent systems with a non-compensatory metric.",
                       help="Topic seed for the discussion (Agent GT only)")
    parser.add_argument("-q", "--query", type=str, default=None,
                       help="Direct query for external GT (required with --gt)")
    parser.add_argument("-g", "--gt", type=str, default=None,
                       help="Ground truth statement for external query (required with --query)")
    parser.add_argument("-i", "--speaker_index", type=int, default=0,
                       help="Index of the initial speaker agent (Agent GT only)")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                       help="Output directory for results (default: creates timestamped folder)")
    parser.add_argument("--include_embeddings", action="store_true",
                       help="Include embeddings in saved results (creates larger files)")
    parser.add_argument("--no_plots", action="store_true",
                       help="Skip displaying plots")
    parser.add_argument("--hyp_djs", action="store_true",
                       help="Use hypothesis-set based Jensen-Shannon divergence for cleaner information-fidelity signals")
    parser.add_argument("--hyp_temp", type=float, default=0.4,
                       help="Softmax temperature for hypothesis-based JSD (lower -> more peaked beliefs, tends to lower u)")

    parser.add_argument("-v", "--verbose", action="store_true", default=VERBOSE,
                       help="Enable verbose output")
    VERBOSE = parser.parse_args().verbose

    return parser.parse_args()

# ----------------------------
# Example main
# ----------------------------

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Determine experiment case based on provided arguments
    if args.query and args.gt:
        case = CASE_B
    elif not args.query and not args.gt:
        case = CASE_A
    else:
        # Error: partial GT arguments provided
        print("Error: Both --query and --gt must be specified together for GT case")
        print("       For agent-to-agent case (Agent), omit both --query and --gt")
        sys.exit(1)

    # Parse agents configuration from environment
    agent_models = AGENTS_CONFIG.split(',')
    agent_models = [model.strip() for model in agent_models]  # Clean whitespace

    # Generate agent names based on number of models
    names = [f"A{i+1}" for i in range(len(agent_models))]
    models = agent_models

    print(f"Running {case} experiment with {len(names)} agents:")
    print(f"  Agent models: {models}")
    print(f"  Max turns: {args.max_turns}")
    print(f"  Min turns: {args.min_turns}")
    print(f"  Tau: {args.tau}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Use local: {args.use_local}")
    print(f"  Use kappa: {args.use_kappa}")
    print(f"  Hypothesis-set JSD: {args.hyp_djs}")
    if case == CASE_B:
        print(f"  Query: {args.query}")
        print(f"  GT Statement: {args.gt}")
    print()

    # Run the appropriate experiment case
    if case == CASE_B:
        results = run_case_GT(
            seed_query=args.query,
            gt_statement=args.gt,
            agent_names=names,
            agent_models=models,
            max_turns=args.max_turns,
            min_turns=args.min_turns,
            tau=args.tau,
            eps=args.epsilon,
            use_local=args.use_local,
            temperature=args.temperature,
            use_kappa=args.use_kappa,
            schedule=args.schedule,
            convergence_variance_threshold=args.convergence_variance_threshold,
            convergence_improvement_threshold=args.convergence_improvement_threshold,
            hyp_djs=getattr(args, 'hyp_djs', False),
            hyp_temperature=getattr(args, 'hyp_temp', 1.0),
        )
    else:  # Case A
        results = run_case_A(
            topic_seed=args.topic_seed,
            agent_names=names,
            agent_models=models,
            speaker_index=args.speaker_index,
            max_turns=args.max_turns,
            min_turns=args.min_turns,
            tau=args.tau,
            eps=args.epsilon,
            use_local=args.use_local,
            temperature=args.temperature,
            use_kappa=args.use_kappa,
            schedule=args.schedule,
            convergence_variance_threshold=args.convergence_variance_threshold,
            convergence_improvement_threshold=args.convergence_improvement_threshold,
            hyp_djs=getattr(args, 'hyp_djs', False),
            hyp_temperature=getattr(args, 'hyp_temp', 1.0),
        )

    # Display results
    if case == CASE_B:
        print("\n=== Query ===")
        print(results["seed_query"])
        print("\n=== Ground Truth Statement ===")
        print(results["gt_statement"])
        print(f"\n=== Final GT Discussion-level score ===")
        print(f"U_discussion_GT (final) = {results['U_discussion']:.4f}")

        print("\n=== Per-turn GT metrics ===")
        display_df = results["per_turn_df"][["turn", "questioner", "answerer", "U_group_turn_gt", "U_discussion_cumulative_gt"]].copy()
        display_df.columns = ["Turn", "Questioner", "Answerer", "U_group^(n)_GT", "U_discussion_GT"]
        print(display_df.to_string(index=False, float_format="%.4f"))

        print(f"\n=== GT Similarity Matrix (final turn) ===")
        for i, name in enumerate(results["original_agent_names"]):
            gt_sim = results["gt_similarity_matrix"][i, 0]
            print(f"{name} -> GT: {gt_sim:.4f}")

    else:  # Case A
        print("\n=== Seed query ===")
        print(results["seed_query"])
        print("\n=== Final Discussion-level score ===")
        print(f"U_discussion (final) = {results['U_discussion']:.4f}")

        print("\n=== Per-turn metrics ===")
        display_df = results["per_turn_df"][["turn", "questioner", "questioner_model", "U_group_turn", "U_discussion_cumulative"]].copy()
        display_df.columns = ["Turn", "Questioner", "Model", "U_group^(n)", "U_discussion (cumulative)"]
        print(display_df.to_string(index=False, float_format="%.4f"))

    print(f"\n=== Q&A Conversation ===")
    for msg in results["qa_conversation"]:
        print(f"  {msg}")

    print(f"\n=== Final agent understanding statements ===")
    # Use original agent names/models for understanding statements (don't include GT)
    if "original_agent_names" in results:  # Case GT
        display_names = results["original_agent_names"]
        display_models = results["original_agent_models"]
    else:  # Case A
        display_names = results["agent_names"]
        display_models = results["agent_models"]

    for i, (name, model, understanding) in enumerate(zip(display_names, display_models, results["understanding_statements"])):
        print(f"{name} ({model}): {understanding}")

    # Save complete results (JSON + plots) to folder
    output_dir = save_experiment_complete(
        results=results,
        local_llm=args.use_local,
        output_dir=args.output_dir,
        include_embeddings=args.include_embeddings,
        save_plots=True,
        show_plots=not args.no_plots
    )

    print(f"\nAll results saved to folder: {output_dir}")
    print("  - results.json (experiment data)")
    print("  - understanding_metrics.png (dual subplot: per-turn + cumulative)")
    print("  - combined_metrics.png (single plot with both metrics)")
    print("  - pairwise_heatmap.png (final understanding matrix)")
    print("  - pairwise_evolution.png (turn-by-turn pairwise matrices starting from turn 2)")
