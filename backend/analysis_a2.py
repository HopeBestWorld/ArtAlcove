from typing import List, Tuple, Dict
from collections.abc import Callable
import numpy as np
import helpers_a2 as helpers
from collections import Counter
import re


def create_j_sim_mat(input_n_speakers: int, input_word_matrix: np.ndarray, input_good_types: List[str]) -> np.ndarray:
    """Create Jaccard similarity matrix for characters.
    Create Jaccard similarity matrix, a np.ndarray of size (`input_n_speakers`, `input_n_speakers`),
    computing the character similarity, where the entry (i, j) indicating the Jaccard similarity
    between the speakers `i` and `j`.
    Hint: To help you out, here are a few numpy operations you might find useful: np.zeros, 
    np.count_nonzero, np.sum.
    Parameters
    ----------
    input_n_speakers : int
        The number of input speakers to be chosen in constructing the Jaccard similarity matrix.
    input_word_matrix : np.ndarray
        The word occurrence matrix of (`n_good_speakers`, `n_good_types`), with the entry (i, j)
        indicating how often the speaker `i` utters word `j`.
    input_good_types : list
        An alphabetically-sorted list of all the words that appear in more than one episode 
        (referred to as "good types" in assignment 1).
    Returns
    -------
    np.ndarray
        The Jaccard similarity matrix of (`input_n_speakers`, `input_n_speakers`), with the entry
        (i, j) indicating the Jaccard similarity between the speakers `i` and `j`.
    """
    # TODO-1.1
    j_matrix = np.zeros((input_n_speakers, input_n_speakers))
    
    for i in range(input_n_speakers):
        for j in range(i, input_n_speakers):  
            inter = np.count_nonzero((input_word_matrix[i, :] > 0) & (input_word_matrix[j, :] > 0))
            uni = np.count_nonzero((input_word_matrix[i, :] > 0) | (input_word_matrix[j, :] > 0))
            
            if uni != 0:
                j_matrix[i, j] = inter / uni
            else:
                j_matrix[i, j] = 0  
            
            j_matrix[j, i] = j_matrix[i, j]
    
    return j_matrix


def avg_sim_dict(input_sim_matrix: np.ndarray, input_good_speakers: List[str]) -> Dict[str, float]:
    """A dictionary of average Jaccard similarity scores.

    Returns a dictionary with the keys being speakers and the values being that character's average 
    similarity scores with all other characters (i.e., ignoring the speaker's self-similarity
    measurement).

    Hint: To help you out, here are a few numpy operations that might be useful: np.sum, np.average.

    Parameters
    ----------
    input_sim_matrix : np.ndarray
        The input similarity matrix, ordered by `input_good_speakers`.
    input_good_speakers : list
        A list of chosen "good" speakers.

    Returns
    -------
    dict
        A dictionary with speakers as keys and associated average similarity score as values.
    """
    # TODO-1.2
    avg_sim_dict = {}

    for i in range(len(input_good_speakers)):
        sim = np.delete(input_sim_matrix[i], i)
        
        avg_s = np.average(sim)
        
        avg_sim_dict[input_good_speakers[i]] = avg_s
    
    return avg_sim_dict


def most_least_unique_characters(input_avg_sim_dict: Dict[str, float]) -> Tuple[Tuple, Tuple]:
    """Returns a tuple which shows the most and least unique characters. 
    The desired tuple format should be the following: 
    ```((MOST_UNIQUE_CHARACTER_NAME, LEAST_SIM_SCORE),
        (LEAST_UNIQUE_CHARACTER_NAME, MOST_SIM_SCORE))```.

    Parameters
    ----------
    input_avg_sim_dict : dict
        A dictionary with speakers as keys and associated average similarity score as values.

    Returns
    -------
    tuple
        A tuple which shows the most and least unique characters, and their similarity scores.
    """
    # TODO-1.3
    most_unique_n = min(input_avg_sim_dict, key=input_avg_sim_dict.get)
    most_unique_s = input_avg_sim_dict[most_unique_n]

    least_unique_n = max(input_avg_sim_dict, key=input_avg_sim_dict.get)
    least_unique_s = input_avg_sim_dict[least_unique_n]

    return ((most_unique_n, most_unique_s), (least_unique_n, least_unique_s))


def tf(word_w: str, character_c: int, input_word_matrix: np.ndarray) -> float:
    """Compute the term frequency weight for a given word and character.

    This function determines the term frequency weight as the ratio between the number of times an 
    input character `character_c` utters word `word_w`, and the total number of words said by 
    character `character_c`.

    Note: Please use full precision in your answer (do NOT round). Also, be sure to use the global
    variable `good_types_reverse_index` from helpers.py (in this method (to ensure a smaller arg list). Finally, you
    may realize that this TF compute is different from the unnormalized TF taught in the class.

    Hint: You may find it helpful to use np.sum.

    Parameters
    ----------
    word_w : str
        The word whose term frequency weight is to be computed.
    character_c : int
        The index of the character within `good_speakers`. 
    input_word_matrix : np.ndarray
        The word occurrence matrix of (`n_good_speakers`, `n_good_types`), with the entry (i, j)
        indicating how often the speaker `i` utters word `j`.

    Returns
    -------
    float
        The (unrounded) term frequency weight for a given word and character.
    """
    # TODO-2.1
    index_word = helpers.good_types_reverse_index[word_w]

    count_word = input_word_matrix[character_c, index_word]

    spoken_total_words = np.sum(input_word_matrix[character_c])

    return count_word / spoken_total_words


def create_g_j_sim_mat(tf_method: Callable[[str, int, np.ndarray], float], input_word_matrix: np.ndarray, input_good_types: List[str], input_n_speakers: int) -> np.ndarray:
    """Create generalized Jaccard similarity matrix for characters.
    Create generalized Jaccard similarity matrix, an np.ndarray of size (`input_n_speakers`, 
    `input_n_speakers), computing the character similarity, where the entry (i, j) indicating the 
    generalized Jaccard similarity between the speakers `i` and `j`.
    Note: Try to minimize the use of `tf_method`.
    Parameters
    ----------
    tf_method : function
        A method to compute the term frequency weight for a given word and character. 
    input_word_matrix : np.ndarray
        The word occurrence matrix of (`n_good_speakers`, `n_good_types`), with the entry (i, j)
        indicating how often the speaker `i` utters word `j`.
    input_good_types : list
        An alphabetically-sorted list of all the words that appear in more than one episode 
        (referred to as "good types" in assignment 1).
    input_n_speakers : int
        The number of input speakers to be chosen in constructing the generalized Jaccard similarity
        matrix.
    Returns
    -------
    np.ndarray
        The generalized Jaccard similarity matrix of (`input_n_speakers`, `input_n_speakers`), with 
        the entry (i, j) indicating the generalized Jaccard similarity between the speakers `i` and
        `j`.
    """
    # TODO-2.2
    g_j_sim_mat = np.zeros((input_n_speakers, input_n_speakers))

    for i in range(input_n_speakers):
        for j in range(i, input_n_speakers):  
            n = 0
            d = 0
            
            for word in input_good_types:
                i_tf = tf_method(word, i, input_word_matrix)
                j_tf = tf_method(word, j, input_word_matrix)
                
                n += min(i_tf, j_tf)
                d += max(i_tf, j_tf)
            
            g_j_sim = n / d if d != 0 else 0.0
            
            g_j_sim_mat[i, j] = g_j_sim
            g_j_sim_mat[j, i] = g_j_sim
    
    return g_j_sim_mat

# incorrect
def create_avg_sims(input_g_sim: np.ndarray) -> np.ndarray:
    """Compute average similarity between a speaker and others, for all speakers.

    Returns a np.ndarray (of size equal to the number of input speakers, `input_n_speakers`) that
    gives the average (mean) similarity between speaker `i` and everybody except the speaker itself.

    Hint: To help you out, here are a few numpy operations that might be useful: np.sum, np.average.

    Parameters
    ----------
    input_g_sim : np.ndarray
        The generalized Jaccard similarity matrix of (`input_n_speakers`, `input_n_speakers`), with 
        the entry (i, j) indicating the generalized Jaccard similarity between the speakers `i` and
        `j`.

    Returns
    -------
    np.ndarray
        An array that gives the average (mean) similarity between speaker `i` and everybody except 
        the speaker itself, for all the speakers.
    """
    # TODO-3.1
    np.fill_diagonal(input_g_sim, 0)
    
    avg_sims = np.sum(input_g_sim, axis=1) / (input_g_sim.shape[1] - 1)
    
    return avg_sims


def most_least_char_sim(input_avg_sims: np.ndarray, input_good_speakers: List[str]) -> Tuple[str, str]:
    """Identify the most and least similar characters.

    Returns a tuple that reveals the most and least similar characters, relative to the rest of the
    characters in `input_good_speakers`. 
    The desired tuple format should be the following:
    ```(MOST_SIMILAR_CHARACTER_NAME, LEAST_SIMILAR_CHARACTER_NAME)```.

    Parameters
    ----------
    input_avg_sims : np.ndarray
        An array that gives the average (mean) similarity between speaker `i` and everybody except 
        the speaker itself, for all the speakers.
    input_good_speakers : list
        A list of chosen "good" speakers.

    Returns
    -------
    tuple
        A tuple of most and least similar characters, relative to the rest of the characters. 
    """
    # TODO-3.2
    idx_most_sim = np.argmax(input_avg_sims)
    c_most_sim = input_good_speakers[idx_most_sim]
    
    i_least_sim = np.argmin(input_avg_sims)
    c_least_sim = input_good_speakers[i_least_sim]
    
    return (c_most_sim, c_least_sim)


def create_reply_matrix(input_deduped_transcripts: List[Tuple], input_good_speakers: List[str], input_n_speakers: int) -> np.ndarray:
    """Create a reply matrix for character interactions.

    Returns a np.ndarray of shape (`input_n_speakers`, `input_n_speakers`) such that an entry (i, j)
    indicates the number of times speaker `j` replied to speaker `i`.

    Parameters
    ----------
    input_deduped_transcripts : list
        The list of input deduped transcripts, loaded from assignment 1.
    input_good_speakers : list
        A list of chosen "good" speakers.
    input_n_speakers : int
        The number of input speakers to be chosen in constructing the reply matrix (essentially, the
        number of good speakers).

    Returns
    -------
    np.ndarray
        The reply matrix of shape (`input_n_speakers`, `input_n_speakers`) such that an entry (i, j)
        indicates the number of times speaker `j` replied to speaker `i`.
    """
    # TODO-4
    reply_matrix = np.zeros((len(input_good_speakers), len(input_good_speakers)))

    i_speaker = {speaker: idx for idx, speaker in enumerate(input_good_speakers)}

    for _, t in input_deduped_transcripts:
        for i in range(len(t) - 1):
            A_speaker = t[i]['speaker']
            B_speaker = t[i+1]['speaker']
            
            if A_speaker in i_speaker and B_speaker in i_speaker:
                idx_A = i_speaker[A_speaker]
                idx_B = i_speaker[B_speaker]
                
                if idx_A != idx_B:
                    reply_matrix[idx_A, idx_B] += 1

    return reply_matrix

def create_good_pairs(input_reply_matrix: np.ndarray,
                      input_good_speakers: List[str],
                      input_n_speakers: int,
                      min_exchange_messages: int = 350) -> List[Tuple[str, str]]:
    """Create "good pairs" based on the number of messages exchanged.

    Returns a tuple list of good pairs in the following format:
    ```[(CHARACTER_NAME_A, CHARACTER_NAME_B), (CHARACTER_NAME_A, CHARACTER_NAME_C), ...]```.

    Note: Good pairs are bi-directional and should be included only once. That is, if (A, B) is in 
    the list, (B, A) should not be. Names should be ordered alphbetically within each tuple. Also,
    sorting the good pairs need not be handled in this function.

    Parameters
    ----------
    input_reply_matrix : np.ndarray
        The reply matrix of shape (`input_n_speakers`, `input_n_speakers`) such that an entry (i, j)
        indicates the number of times speaker `j` replied to speaker `i`.
    input_good_speakers : list
        A list of chosen "good" speakers.
    input_n_speakers : int
        The number of input speakers to be chosen in constructing the good pairs (essentially, the
        number of good speakers).
    min_exchange_messages : int
        The minimum number of messages to be exchanged between a character pair for them to be 
        considered a "good pair" (defaults to 350).

    Returns
    -------
    list
        A list of tuples of good pairs, based on the number of messages exchanged.
    """
    # TODO-5.1
    good_pairs = []

    for i in range(input_n_speakers):
        for j in range(i + 1, input_n_speakers):
            total_messages = input_reply_matrix[i, j] + input_reply_matrix[j, i]

            if total_messages >= min_exchange_messages:
                pair = tuple(sorted((input_good_speakers[i], input_good_speakers[j])))
                good_pairs.append(pair)

    return good_pairs


def create_weighted_words(input_pair_words_mat: np.ndarray) -> np.ndarray:
    """Create weighted character-pair word occurrence matrix.

    Returns a np.ndarray with the same shape as `input_pair_words_mat`, such that entry (i, j) 
    indicates a weighted score showing how often a given pair replied a word.

    Note: Words may not be said by any pair.
    Hint: Use numpy primitives to optimize this function.

    Parameters
    ----------
    input_pair_words_mat : np.ndarray
        The character-pair word occurrence matrix with an entry (i, j) indicating how many times 
        pair `i` has replied good type word `j`.

    Returns
    -------
    np.ndarray
        A weighted character-pair word occurrence matrix such that entry (i, j) indicates a weighted
        score showing how often a given pair replied a word.
    """
    # TODO-5.3
    smoothed_counts = input_pair_words_mat + 1

    column_sums = np.sum(smoothed_counts, axis=0)

    column_sums = np.where(column_sums == 0, 1, column_sums)

    weighted_words = smoothed_counts / column_sums

    return weighted_words


def create_interaction_mat(input_reply_matrix: np.ndarray,
                           input_n_speakers: int) -> np.ndarray:
    """Create an interaction matrix.

    Returns a np.ndarray of shape (`input_n_speakers`, `input_n_speakers`), such that the entry
    (i, j) indicates the combined number of times speaker `j` started a conversation with speaker
    `i` and vice versa.

    Parameters
    ----------
    input_reply_matrix : np.ndarray
        The reply matrix of shape (`input_n_speakers`, `input_n_speakers`) such that an entry (i, j)
        indicates the number of times speaker `j` replied to speaker `i`.
    input_n_speakers : int
        The number of input speakers to be chosen in constructing the reply matrix.

    Returns
    -------
    np.ndarray
        An interaction matrix, such that the entry (i, j) indicates the combined number of times 
        speaker `j` started a conversation with speaker `i` and vice versa.
    """
    # TODO-6
    interaction_matrix = input_reply_matrix + input_reply_matrix.T
    
    return interaction_matrix

# wrong
def create_age_weighted_words(input_age_words_mat: np.ndarray) -> np.ndarray:
    """Create weighted age-group interactions matrix.

    Returns a np.ndarray with the same shape as `input_age_words_mat`, such that entry (i, j) 
    indicates a weighted score showing how often a given age-group pair `i` uttered a good type
    word `j`.

    Note: Words may not be said by any pair.
    Hint: Use numpy primitives to optimize this function.

    Parameters
    ----------
    input_age_words_mat : np.ndarray
        The age-group interactions matrix with an entry (i, j) indicating how many times age-group 
        pair `i` has uttered good type word `j`.

    Returns
    -------
    np.ndarray
        A weighted age-group interactions matrix with an entry (i, j) indicating how many times age
        group pair `i` has uttered good type word `j`.
    """
    # TODO-7.2
    smoothed_age_words_mat = input_age_words_mat + 1.5

    col_sums = np.sum(smoothed_age_words_mat, axis=0, keepdims=True)
    weighted_age_words_mat = smoothed_age_words_mat / col_sums

    return weighted_age_words_mat


def create_pair_priming_mat(input_tokenize_method: Callable[[str], List[str]],
                            input_ordered_pair_lines: Dict[Tuple[str, str], List[List[Dict[str, str]]]],
                            input_good_speakers: List[str],
                            input_n_speakers: int,
                            ) -> np.ndarray:
    """Create the priming matrix from pair-wise ordered transcript lines.

    Returns a np.ndarray of size (`input_n_speakers`, `input_n_speakers`) where entry (i, j) is the
    mean of priming computations (proportions where numerator is the number of good types in `j`'s 
    reply also said by `i` and denominator is number of good types in `j`'s reply) across all pairs
    of lines where `j` replied to `i`.

    Hint: Use numpy primitives to make this function optimal.
    Note: You should use `good_types_reverse_index` global variable from helpers.py

    Steps:
        1. Iterate through each tuple (for each pair i, j).
        2. Grab only good types.
        3. Do the necessary priming calculations.

    Parameters
    ----------
    input_tokenize_method : function
        A method that tokenizes the input text into tokens.
    input_ordered_pair_lines : dict
        A dictionary of all of the lines and replies between pairs of speakers, organized as a dict
        (keys: pairs of speakers) of list of dict lists (as values);.
    input_good_speakers : list
        A list of chosen "good" speakers.
    input_n_speakers : int
        The number of input speakers to be chosen in constructing the priming matrix.
    Returns
    -------
    np.ndarray
        The priming matrix of size (`input_n_speakers`, `input_n_speakers`), where entry (i, j) is
        the mean of priming computations across all pairs of lines where `j` replied to `i`.
    """

    # We are providing you the code structure below to help you get started in the right direction.
    # You may edit it, if you want.
    good_types_set = set(helpers.good_types_reverse_index.keys())

    # Resultant matricies.
    proportions = np.zeros((input_n_speakers, input_n_speakers))
    totals = np.zeros((input_n_speakers, input_n_speakers))

    # All the pairs.
    for pair in input_ordered_pair_lines.keys():
        A = pair[0]  # A, who is prompting B
        i = input_good_speakers.index(A)
        B = pair[1]  # B, who is responding to A
        j = input_good_speakers.index(B)
        # For tuple arrays for each pair.
        for arrs in input_ordered_pair_lines[pair]:
            # Grab sets of words.
            response = input_tokenize_method(arrs[1]['text'])
            B_sent = set(response)
            A_sent = set(input_tokenize_method(arrs[0]['text']))

            # TODO-8.1:
            # 1. Reduce `B_sent` to just good types.
            # 2. Up the total number of correspondences for each pair.
            # 3. Compute priming calculations and sum for each pair.
            B_good = B_sent.intersection(good_types_set)
        
            if not B_good:
              continue
        
            primed_words = A_sent.intersection(B_good)
            priming_proportion = len(primed_words) / len(B_good)
            
            proportions[i, j] += priming_proportion
            totals[i, j] += 1

    totals[totals == 0] += 1
    answer = proportions / totals
    return answer

# wrong
def word_counts_per_ordered_pair(input_ordered_pairs: List[Tuple[str, str]],
                                 input_ordered_pairs_lines: Dict[Tuple[str, str], List[List[Dict[str, str]]]],
                                 ) -> Dict[Tuple[str, str], int]:
    """Retrieve word counts of primed words per ordered pair.

    Returns a dictionary that contains all the word counts of primed words (`good_types`) per 
    ordered pair in the following format:
    ```{
           (character_index_a, character_index_b): {word_1: count_1, word_2: count_2...},
           (character_index_a, character_index_c): {word_1: count_1, word_2: count_2...},
           ...
        ...
       }```.

    Hint: Make sure to use the `good_types_reverse_index` global variable provided in helpers.py.

    Note: The dictionary you are returning should contain all word counts of primed words 
    (`good_types`), not just the top ten, and for all pairs, not just the top three.

    Parameters
    ----------
    input_ordered_pairs : list
        A list of all ordered pairs of characters with correspondance in `input_ordered_pair_lines`.
    input_ordered_pair_lines : dict
        A dictionary of all of the lines and replies between pairs of speakers, organized as a dict
        (keys: pairs of speakers) of list of dict lists (as values);        
    Returns
    -------
    dict
        A dictionary that contains all the word counts of primed words (`good_types`) per ordered
        pair in the required format.
    """
    # TODO-8.2
    priming_word_counts = {}

    for pair in input_ordered_pairs:
        speaker_a, speaker_b = pair
        word_counter = Counter()

        conversations = input_ordered_pairs_lines.get(pair, [])

        for convo in conversations:
            if len(convo) >= 2:
                a_words = set(re.findall(r'\b\w+\b', convo[0]['text'].lower()))
                b_words = re.findall(r'\b\w+\b', convo[1]['text'].lower())

                repeated_words = [word for word in b_words if word in a_words and word in helpers.good_types_reverse_index]
                word_counter.update(repeated_words)

        priming_word_counts[pair] = dict(word_counter)

    return priming_word_counts