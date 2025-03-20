import re
import matplotlib.pyplot as plt
import numpy as np

# Tokenize method which can be passed into various methods.
def tokenize(text):
    """Returns a list of words that make up the text.    

    Parameters
    ----------
    text : str
        The input text string

    Returns
    -------
    list
        A list of tokens corresponding to the input string.
    """
    return [x for x in re.findall(r"[a-z]+", text.lower())]





# Helper methods and variables for [PART 2][8]
def merge_deduped(transcripts):
    """Merges adjacent transcript lines by the same speaker."""
    result = []
    for t_id, tscript in transcripts:
        prev_speaker = tscript[0]['speaker']
        prev_line = tscript[0]['text']
        tscript_result = []
        for i in range(1, len(tscript)):
            curr_speaker = tscript[i]['speaker']
            curr_line = tscript[i]['text']
            if curr_speaker == prev_speaker:
                prev_line = prev_line + " " + curr_line
            else:
                tscript_result.append({'speaker': prev_speaker, 'text': prev_line})
                prev_speaker = curr_speaker
                prev_line = curr_line
            if i == len(tscript) - 1:
                tscript_result.append({'speaker': prev_speaker, 'text': prev_line})
        result.append((t_id, tscript_result))
    return result


def prev_lines_and_replies(A, B, tscript):
    """
    Creates an array of all the transcript lines involving supposed correspondence between A (first 
    speaker) and B (second speaker), both previous message and current message.
    """
    # Prepare results for B responses to A, previous speak.
    result = []
    prev_speaker = None
    prev_line = None

    # Go through every line of the transcript.
    for line in tscript:
        curr_speaker = line['speaker']

        # If dialogue has happened.
        if ((curr_speaker == B and prev_speaker == A)):
            # Add the previous line and the current line.
            result.append([prev_line, line])

        # Update the speaker and line.
        prev_speaker = curr_speaker
        prev_line = line

    return result