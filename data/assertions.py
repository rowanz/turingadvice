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