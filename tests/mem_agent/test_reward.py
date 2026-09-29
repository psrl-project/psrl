from examples.mem_agent.reward import compute_score, extract_boxed_answer


def test_extract_boxed_answer_uses_last_box():
    assert extract_boxed_answer("draft \\boxed{wrong}; final \\boxed{right}") == "right"


def test_reward_accepts_any_ground_truth_answer():
    result = compute_score(
        "mem_agent_hotpotqa",
        "The answer is \\boxed{New York City}",
        ["NYC", "new york city"],
        {"num_turns": 1},
    )
    assert result == {
        "score": 1.0,
        "acc": 1.0,
        "f1": 1.0,
        "em": 1.0,
        "sub_em": 1.0,
    }


def test_reward_requires_boxed_answer():
    assert compute_score("mem_agent_hotpotqa", "New York City", ["New York City"], {"num_turns": 1})["score"] == 0.0
