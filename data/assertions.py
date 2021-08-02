def question_is_valid(question: dict) -> bool:
    if "selftext" in question:
        return len(question["selftext"]) >= 64
    else:
        return False

def answer_is_valid(answer: dict) -> bool:
    if "body" in answer:
        return len(answer["body"]) >= 64
    else:
        return False

def answer_pair_is_valid(
    ans1: dict, ans2: dict, max_time_diff: int, max_len_ratio: float,
    min_score_ratio: float
    ):
    time_diff = abs(ans1["created_utc"] - ans2["created_utc"])
    ans_lengths = [len(ans1["body"]), len(ans2["body"])]
    len_ratio = max(ans_lengths) / min(ans_lengths)
    ans_scores = [ans1["score"], ans2["score"]]
    score_ratio = max(ans_scores) / min(ans_scores)
    return (
        time_diff <= max_time_diff \
        and len_ratio <= max_len_ratio \
        and score_ratio >= min_score_ratio \
        and ans1["score"] != ans2["score"]
    )