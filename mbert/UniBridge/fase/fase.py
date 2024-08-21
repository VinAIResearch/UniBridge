from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from .fasttext_embs import load_target_token_embedding


BPE_WHITESPACE = "Ġ"
XLMR_WHITESPACE = "▁"


def get_token_standardization_func(input_tokenizer: PreTrainedTokenizerFast):
    """Standardize tokens from different tokenizers.
    Standard output format should be Unicode-like output for non-ASCII chars.
    Beginning of word tokens should be prefixed with a space.

    We have to use .decode() to get "standardized" tokens (e.g. BytePairBPE represents non-ASCII tokens non-UNIcode-like internally).
    But XLM-R's tokenizer removes leading whitespace from tokens when using .decode().
    Se we add those back in manually.
    """

    def decode(tokenizer: PreTrainedTokenizerFast, token_id: int):
        """For BPE tokenizer and fallback"""
        return tokenizer.decode(token_id)

    def replace_space(tokenizer: PreTrainedTokenizerFast, token_id: int):
        """For XLM-R tokenizer (sentencepiece-style)"""
        return tokenizer.convert_ids_to_tokens(token_id).replace(XLMR_WHITESPACE, " ")

    def wordpiece(tokenizer: PreTrainedTokenizerFast, token_id: int):
        """For wordpiece (e.g. BERT or mBERT)"""
        token = tokenizer.decode(token_id)
        if token.startswith("##"):
            return token[2:]
        else:
            return " " + token

    # simple heuristics to avoid false positive
    if len([k for k in input_tokenizer.get_vocab().keys() if k[0] == XLMR_WHITESPACE]) > 100:
        standardize_token = replace_space
    # simple heuristics to avoid false positive
    elif len([k for k in input_tokenizer.get_vocab().keys() if k[:2] == "##"]) > 100:
        standardize_token = wordpiece
    else:
        standardize_token = decode

    return standardize_token


def get_overlapping_tokens(
    target_tokenizer: PreTrainedTokenizerFast,
    source_tokenizer: PreTrainedTokenizerFast,
    fuzzy_search=True,
    fuzzy_whitespace=False,
):
    target_vocab = target_tokenizer.get_vocab()
    source_vocab = source_tokenizer.get_vocab()

    standardize_token = get_token_standardization_func(source_tokenizer)
    source_vocab = {standardize_token(source_tokenizer, idx): idx for idx in sorted(source_vocab.values())}

    standardize_token = get_token_standardization_func(target_tokenizer)
    target_vocab = {standardize_token(target_tokenizer, idx): idx for idx in sorted(target_vocab.values())}

    # Determine overlapping tokens between source and target vocab
    exact_overlap = {k: (target_vocab[k], source_vocab[k]) for k in set(target_vocab) & set(source_vocab)}

    if not fuzzy_search:
        return {target_tokenizer.convert_ids_to_tokens(v[0]): v for k, v in sorted(exact_overlap.items())}

    # We do a greedy search for additional overlapping tokens.
    # NOTE: source_vocab order is random, need to sort for consistent results
    lowercase_source_vocab = {k.lower(): v for k, v in sorted(source_vocab.items())}
    fuzzy_overlap = exact_overlap

    for target_token, target_token_idx in sorted(target_vocab.items()):
        lowercase_target_token = target_token.lower()
        if fuzzy_overlap.get(target_token):
            continue
        if lowercase_source_vocab.get(lowercase_target_token):
            # same token but just different case found in source vocab
            fuzzy_overlap[target_token] = (
                target_token_idx,
                lowercase_source_vocab[lowercase_target_token],
            )
        elif fuzzy_whitespace and lowercase_source_vocab.get(" " + lowercase_target_token):
            # same token with extra whitespace found in source vocab
            fuzzy_overlap[target_token] = (
                target_token_idx,
                lowercase_source_vocab[" " + lowercase_target_token],
            )
        elif fuzzy_whitespace and lowercase_source_vocab.get(lowercase_target_token.lstrip()):
            # same token without extra whitespace found in source vocab
            fuzzy_overlap[target_token] = (
                target_token_idx,
                lowercase_source_vocab[lowercase_target_token.lstrip()],
            )
    return {target_tokenizer.convert_ids_to_tokens(v[0]): v for k, v in fuzzy_overlap.items()}


@torch.no_grad()
def fase_additional_token_initialization(
    fasttext_model,
    shared_tokens,
    new_tokens,
    target_embeddings: Tensor,
    p=1.0,
    temperature=1,
):
    def sanitized_fasttext_vector(token, fasttext_model):
        """
        Some tokens are not part of fasttext model even though they are in the target tokenizer vocab.
        Calling fasttext_model[<token>] will return combination of subword ngrams for OOV <token>.
        However, when the OOV token is short (e.g. 1 letter), there might be none and a zero-vector will be returned.
        This is bad, because a zero-vector leads to NAN in cosine similarity (division by zero).
        """
        ftv = fasttext_model[token]
        if sum(ftv) == 0:
            ftv = np.random.randn(*ftv.shape)
        return ftv

    new_token_fasttext_embs = OrderedDict(
        ((token, sanitized_fasttext_vector(token, fasttext_model)) for token in new_tokens.keys())
    )
    shared_token_fasttext_embs = OrderedDict(
        ((token, sanitized_fasttext_vector(token, fasttext_model)) for token in shared_tokens.keys())
    )

    new_token_ft_emb_matrix = np.asarray([t for t in list(new_token_fasttext_embs.values())], dtype="float32")
    shared_token_ft_emb_matrix = np.asarray([t for t in list(shared_token_fasttext_embs.values())], dtype="float32")

    from fastdist import fastdist

    new_to_shared_cosine_sims = fastdist.cosine_matrix_to_matrix(new_token_ft_emb_matrix, shared_token_ft_emb_matrix)

    shared_token_idx_to_target_vocab = list(shared_token_fasttext_embs.keys())
    for new_token, shared_token_cosine_sims in tqdm(
        zip(list(new_token_fasttext_embs.keys()), new_to_shared_cosine_sims),
        desc="FASE initialization...",
        total=len(new_to_shared_cosine_sims),
    ):
        ranked_shared_token_idxs = np.argsort(shared_token_cosine_sims)[::-1]
        ranked_shared_token_embs = np.sort(shared_token_cosine_sims)[::-1]

        import entmax

        sparsemax = entmax.sparsemax(torch.from_numpy(ranked_shared_token_embs.copy()) / temperature).numpy()

        accumulated_prob_mass = 0.0
        convex_combination = torch.zeros_like(target_embeddings[0])
        for sparsemax_prob_mass, ranked_shared_token_idx in zip(sparsemax, ranked_shared_token_idxs):
            if sparsemax_prob_mass == 0.0 or accumulated_prob_mass >= p:
                break
            ranked_shared_token_idx_in_target_vocab = shared_tokens[
                shared_token_idx_to_target_vocab[ranked_shared_token_idx]
            ]
            convex_combination += sparsemax_prob_mass * target_embeddings[ranked_shared_token_idx_in_target_vocab]
            accumulated_prob_mass += sparsemax_prob_mass

        # scale all coefficients s.t. we have convex combination (sum of all coefficients is 1 and each coeffcient is > 0)
        # post-hoc here because it's easier to implement
        # in case of p threshold == 1, this is a no-op
        if p < 1.0:
            convex_combination = convex_combination / accumulated_prob_mass

        # Initialize the new token embedding with the FASE combination
        target_embedding_idx = new_tokens[new_token]
        target_embeddings[target_embedding_idx] = convex_combination

    return target_embeddings.detach()


@torch.inference_mode()
def get_semantic_sim_tokens(
    src_fasttext_model, tgt_fasttext_model, additional_tokens, source_tokenizer, target_tokenizer
):
    def sanitized_fasttext_vector(token, fasttext_model):
        """
        Some tokens are not part of fasttext model even though they are in the target tokenizer vocab.
        Calling fasttext_model[<token>] will return combination of subword ngrams for OOV <token>.
        However, when the OOV token is short (e.g. 1 letter), there might be none and a zero-vector will be returned.
        This is bad, because a zero-vector leads to NAN in cosine similarity (division by zero).
        """
        ftv = fasttext_model[token]
        if sum(ftv) == 0:
            ftv = np.random.randn(*ftv.shape)
        return ftv

    non_lex_sim_fasttext_embs = OrderedDict(
        ((token, sanitized_fasttext_vector(token, tgt_fasttext_model)) for token in additional_tokens.keys())
    )

    source_vocab = source_tokenizer.get_vocab()
    standardize_token = get_token_standardization_func(source_tokenizer)
    source_vocab = {standardize_token(source_tokenizer, idx): idx for idx in sorted(source_vocab.values())}

    src_fasttext_embs = OrderedDict(
        ((token, sanitized_fasttext_vector(token, src_fasttext_model)) for token in source_vocab.keys())
    )

    tgt_embeds = list(non_lex_sim_fasttext_embs.values())
    tgt_tokens = list(additional_tokens.keys())

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    mat_tgt = torch.from_numpy(np.asarray([t for t in tgt_embeds], dtype="float32")).to(device)
    mat_src = torch.from_numpy(np.asarray([t for t in list(src_fasttext_embs.values())], dtype="float32")).to(device)

    overlapping_token = {}
    for tgt_idx, embed_tgt in tqdm(enumerate(tgt_embeds)):
        tensor_embed_tgt = torch.tensor(
            np.asarray([embed_tgt]), dtype=torch.float32, device=mat_src.device
        )  # [1, 300]
        dist = F.cosine_similarity(tensor_embed_tgt, mat_src)
        sim_idx_src = torch.argmax(dist, dim=0)
        dist = F.cosine_similarity(mat_src[sim_idx_src, :].unsqueeze(0), mat_tgt)
        sim_idx_tgt = torch.argmax(dist, dim=0)
        if sim_idx_tgt == tgt_idx:
            # match semantic alignment
            token = tgt_tokens[tgt_idx]
            overlapping_token[token] = (additional_tokens[token], sim_idx_src)

    return overlapping_token


def FASE(
    target_tokenizer: PreTrainedTokenizerFast,
    source_tokenizer: PreTrainedTokenizerFast,
    source_embeddings: Tensor,
    target_training_data_path: Optional[str] = None,
    fasttext_model_path: Optional[str] = None,
    language_identifier: Optional[str] = None,
    fasttext_epochs: int = 3,
    fasttext_embedding_dim: int = 300,
    processes: Optional[int] = None,
    debug: bool = False,
) -> Tensor:
    src_fasttext_model = load_target_token_embedding(
        target_tokenizer=source_tokenizer,
        target_training_data_path=target_training_data_path,
        language_identifier=language_identifier,
        fasttext_model_path=fasttext_model_path,
        fasttext_epochs=fasttext_epochs,
        fasttext_embedding_dim=fasttext_embedding_dim,
        processes=processes,
    )

    fasttext_model = load_target_token_embedding(
        target_tokenizer=target_tokenizer,
        target_training_data_path=target_training_data_path,
        language_identifier=language_identifier,
        fasttext_model_path=fasttext_model_path,
        fasttext_epochs=fasttext_epochs,
        fasttext_embedding_dim=fasttext_embedding_dim,
        processes=processes,
    )

    target_token_set = set(target_tokenizer.get_vocab().keys())

    if isinstance(fasttext_model, dict):
        missing_tokens = target_token_set.difference(set(fasttext_model.keys()))
    else:
        missing_tokens = target_token_set.difference(set(fasttext_model.words))

    if debug and len(missing_tokens) > 0:
        logger.warning(
            f"{len(missing_tokens)} target tokens not in fasttext model: {missing_tokens}.  Note: a small number is okay."
        )

    overlapping_token_mapping = get_overlapping_tokens(target_tokenizer, source_tokenizer, fuzzy_search=True)

    target_embeddings = torch.zeros((len(target_tokenizer), source_embeddings.shape[1]))

    # Copy embeddings for overlapping tokens
    # lexical similar
    overlapping_tokens = {}
    for overlapping_token, (
        target_vocab_idx,
        source_vocab_idx,
    ) in overlapping_token_mapping.items():
        overlapping_tokens[overlapping_token] = target_vocab_idx
        target_embeddings[target_vocab_idx] = source_embeddings[source_vocab_idx]

    logger.success(f"Accumulate {len(overlapping_tokens)} lexical overlapped token")

    additional_tokens = {
        token: idx for token, idx in target_tokenizer.get_vocab().items() if not overlapping_tokens.get(token)
    }

    # semantic similar
    semantic_aligned_tokens = get_semantic_sim_tokens(
        src_fasttext_model, fasttext_model, additional_tokens, source_tokenizer, target_tokenizer
    )
    for overlapping_token, (
        target_vocab_idx,
        source_vocab_idx,
    ) in semantic_aligned_tokens.items():
        overlapping_tokens[overlapping_token] = target_vocab_idx
        target_embeddings[target_vocab_idx] = source_embeddings[source_vocab_idx]

    logger.success(f"Accumulate {len(semantic_aligned_tokens)} semantical aligned token")

    additional_tokens = {
        token: idx for token, idx in target_tokenizer.get_vocab().items() if not overlapping_tokens.get(token)
    }

    target_embeddings = fase_additional_token_initialization(
        fasttext_model, overlapping_tokens, additional_tokens, target_embeddings
    )

    return target_embeddings
