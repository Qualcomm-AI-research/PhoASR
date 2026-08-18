# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Token mapping helpers for projecting normalized text back onto the original surface."""

from __future__ import annotations

import string


def longest_common_subsequence(x_list: list[str], y_list: list[str]) -> list[str]:
    """Return the case-insensitive LCS between original and normalized tokens."""
    m, n = len(x_list), len(y_list)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if x_list[i] == y_list[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])

    lcs: list[str] = []
    i, j = m, n
    while i > 0 and j > 0:
        if x_list[i - 1].lower() == y_list[j - 1].lower():
            lcs.append(x_list[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    lcs.reverse()
    return lcs


def are_similar_words(word1: str, word2: str) -> bool:
    """Return whether two tokens differ only by punctuation or casing."""
    word1_clean = "".join(c for c in word1.lower() if c.isalnum())
    word2_clean = "".join(c for c in word2.lower() if c.isalnum())
    return word1_clean == word2_clean


def create_mapping(x_list: list[str], y_list: list[str], lcs: list[str]) -> list[tuple[str, str]]:
    """Map original tokens to normalized tokens using the token-diff rules."""
    mapping: list[tuple[str, str]] = []
    lx, ly = len(x_list), len(y_list)
    lcs_index = 0
    x_index, y_index = 0, 0

    while x_index < lx or y_index < ly:
        if (
            lcs_index < len(lcs)
            and x_index < lx
            and y_index < ly
            and x_list[x_index].lower() == lcs[lcs_index].lower()
            and y_list[y_index].lower() == lcs[lcs_index].lower()
        ):
            mapping.append((x_list[x_index], y_list[y_index]))
            x_index += 1
            y_index += 1
            lcs_index += 1
        else:
            x_diff: list[str] = []
            y_diff: list[str] = []

            while x_index < lx and (
                lcs_index >= len(lcs) or x_list[x_index].lower() != lcs[lcs_index].lower()
            ):
                x_diff.append(x_list[x_index])
                x_index += 1

            while y_index < ly and (
                lcs_index >= len(lcs) or y_list[y_index].lower() != lcs[lcs_index].lower()
            ):
                y_diff.append(y_list[y_index])
                y_index += 1

            if x_diff and y_diff:
                if len(x_diff) == len(y_diff):
                    for i in range(len(x_diff)):
                        mapping.append((x_diff[i], y_diff[i]))
                elif len(x_diff) == 1 and len(y_diff) > 1:
                    mapping.append((x_diff[0], " ".join(y_diff)))
                elif len(x_diff) > 1 and len(y_diff) == 1:
                    mapping.append((" ".join(x_diff), y_diff[0]))
                else:
                    matched_x = [False] * len(x_diff)
                    matched_y = [False] * len(y_diff)
                    for i in range(len(x_diff)):
                        if matched_x[i]:
                            continue
                        for j in range(len(y_diff)):
                            if matched_y[j]:
                                continue
                            if are_similar_words(x_diff[i], y_diff[j]):
                                mapping.append((x_diff[i], y_diff[j]))
                                matched_x[i] = True
                                matched_y[j] = True
                                break

                    remaining_x = [x_diff[i] for i in range(len(x_diff)) if not matched_x[i]]
                    remaining_y = [y_diff[i] for i in range(len(y_diff)) if not matched_y[i]]

                    if remaining_x and remaining_y:
                        if len(remaining_x) == len(remaining_y):
                            for i in range(len(remaining_x)):
                                mapping.append((remaining_x[i], remaining_y[i]))
                        else:
                            mapping.append((" ".join(remaining_x), " ".join(remaining_y)))
                    elif remaining_x:
                        mapping.append((" ".join(remaining_x), ""))
                    elif remaining_y:
                        mapping.append(("", " ".join(remaining_y)))
            elif x_diff:
                mapping.append((" ".join(x_diff), ""))
            elif y_diff:
                mapping.append(("", " ".join(y_diff)))

    combined_mapping: list[tuple[str, str]] = []
    for i in range(len(mapping)):
        if i > 0 and mapping[i][0] == "" and combined_mapping[-1][1] != "":
            combined_mapping[-1] = (
                combined_mapping[-1][0],
                combined_mapping[-1][1] + " " + mapping[i][1],
            )
        else:
            combined_mapping.append(mapping[i])

    fixed_mapping: list[tuple[str, str]] = []
    i = 0
    while i < len(combined_mapping):
        if i + 1 < len(combined_mapping):
            curr_x, curr_y = combined_mapping[i]
            next_x, next_y = combined_mapping[i + 1]
            if curr_y == "" and " " in next_y and any(c.isdigit() for c in curr_x):
                next_y_words = next_y.split()
                last_word = next_y_words[-1]
                remaining_words = " ".join(next_y_words[:-1])
                fixed_mapping.append((curr_x, remaining_words))
                fixed_mapping.append((next_x, last_word))
                i += 2
                continue
        fixed_mapping.append(combined_mapping[i])
        i += 1

    return fixed_mapping


def valid_input(text: str) -> bool:
    """Return whether text contains non-alpha content that needs mapping preservation."""
    for char in text:
        if char.isalpha() or char in string.punctuation or char == " ":
            continue
        return True
    return False


def create_alignment_input(
    text_with_punc: str, text_normalized: str
) -> tuple[list[tuple[str, str]], str, bool]:
    """Build mapping artifacts and the text that should be sent to alignment."""
    x_list = text_with_punc.split()
    y_list = text_normalized.split()
    lcs = longest_common_subsequence(x_list, y_list)
    mappings = create_mapping(x_list, y_list, lcs)
    alignment_input: list[str] = []
    has_empty_mapping = False
    for before, after in mappings:
        if before == "" or after == "":
            has_empty_mapping = True
        if any(char.isdigit() for char in before):
            alignment_input.append(after)
        else:
            alignment_input.append(before)
    return mappings, " ".join(alignment_input).strip(), has_empty_mapping


def aggregate_timestamps(
    timestamps: dict, mappings: list[list[str]] | list[tuple[str, str]]
) -> list[dict]:
    """Project normalized-word timestamps back onto the original token mapping."""
    segments = timestamps["segments"]
    aggregated: list[dict] = []
    i = 0
    for before, after in mappings:
        before_words = before.split()
        after_words = after.split()
        if i >= len(segments):
            break
        if any(c.isdigit() for c in before):
            num_words = len(after_words)
            if i + num_words > len(segments):
                break
            matched_segments = segments[i : i + num_words]
            word_info = {
                "words": [before],
                "start": matched_segments[0]["start"],
                "end": matched_segments[-1]["end"],
                "needs_conversion": True,
            }
            aggregated.append(word_info)
            i += num_words
        elif len(before_words) == 1:
            word_info = {
                "words": [before],
                "start": segments[i]["start"],
                "end": segments[i]["end"],
                "needs_conversion": False,
            }
            aggregated.append(word_info)
            i += 1
        else:
            if i + len(before_words) > len(segments):
                break
            matched_segments = segments[i : i + len(before_words)]
            matched_words = [s["word"] for s in matched_segments]
            if " ".join(matched_words) == before:
                for word, segment in zip(before_words, matched_segments):
                    word_info = {
                        "words": [word],
                        "start": segment["start"],
                        "end": segment["end"],
                        "needs_conversion": False,
                    }
                    aggregated.append(word_info)
                i += len(before_words)
            else:
                word_info = {
                    "words": [segments[i]["word"]],
                    "start": segments[i]["start"],
                    "end": segments[i]["end"],
                    "needs_conversion": False,
                }
                aggregated.append(word_info)
                i += 1
    return aggregated
