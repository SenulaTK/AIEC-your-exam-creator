import re
from typing import Any, Optional
from ..models.exam import Question, GradedQuestion

def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip().lower()

def grade_mcq(q: Question, student_answer: Any, q_idx: int) -> GradedQuestion:
    marks = q.marks
    ans_norm = normalize_text(student_answer)
    correct_norm = normalize_text(q.correct_answer)
    
    if not ans_norm:
        return GradedQuestion(
            question_index=q_idx,
            score=0,
            feedback=f"No answer provided. Correct answer: {q.correct_answer}",
            graded_by="deterministic"
        )

    # 1. Exact normalized match
    if ans_norm == correct_norm:
        return GradedQuestion(
            question_index=q_idx,
            score=marks,
            feedback="Correct! Selected option matches the mark scheme.",
            graded_by="deterministic"
        )
    
    # 2. Check prefix match if answer is like "A) ..." or option text
    if q.options:
        # Check if student gave an index or letter like "A" or 0
        for i, opt in enumerate(q.options):
            opt_norm = normalize_text(opt)
            letter = chr(ord('a') + i)
            if (ans_norm == letter or ans_norm == str(i + 1) or ans_norm == opt_norm) and (opt_norm == correct_norm or correct_norm.startswith(f"{letter})") or correct_norm.startswith(opt_norm)):
                return GradedQuestion(
                    question_index=q_idx,
                    score=marks,
                    feedback="Correct! Selected option matches the mark scheme.",
                    graded_by="deterministic"
                )

    return GradedQuestion(
        question_index=q_idx,
        score=0,
        feedback=f"Incorrect. Your answer: '{student_answer}'. Correct answer: '{q.correct_answer}'",
        graded_by="deterministic"
    )

def grade_true_false(q: Question, student_answer: Any, q_idx: int) -> GradedQuestion:
    marks = q.marks
    ans_norm = normalize_text(student_answer)
    correct_norm = normalize_text(q.correct_answer)

    if not ans_norm:
        return GradedQuestion(
            question_index=q_idx,
            score=0,
            feedback=f"No answer provided. Correct answer is: {q.correct_answer}",
            graded_by="deterministic"
        )

    expected_bool = "true" in correct_norm.split() or correct_norm.startswith("true")
    if "false" in correct_norm.split() or correct_norm.startswith("false"):
        expected_bool = False

    student_bool = ans_norm in ["true", "t", "yes", "y", "1"]
    student_is_false = ans_norm in ["false", "f", "no", "n", "0"]

    if expected_bool and student_bool:
        return GradedQuestion(
            question_index=q_idx,
            score=marks,
            feedback="Correct! Statement is True.",
            graded_by="deterministic"
        )
    elif (not expected_bool) and student_is_false:
        return GradedQuestion(
            question_index=q_idx,
            score=marks,
            feedback=f"Correct! Statement is False. {q.correct_answer}",
            graded_by="deterministic"
        )
    else:
        return GradedQuestion(
            question_index=q_idx,
            score=0,
            feedback=f"Incorrect. Expected {'True' if expected_bool else 'False'}. Reference: {q.correct_answer}",
            graded_by="deterministic"
        )

def grade_matching(q: Question, student_answer: Any, q_idx: int) -> GradedQuestion:
    marks = q.marks
    # Expected format: student_answer is dict {left_item: right_item} or string
    if not student_answer or not isinstance(student_answer, dict):
        return GradedQuestion(
            question_index=q_idx,
            score=0,
            feedback=f"Incomplete or missing matching submission. Correct pairs: {q.correct_answer}",
            graded_by="deterministic"
        )

    # Parse key-value pairs from q.correct_answer if format is "A -> B; C -> D"
    pairs = {}
    for part in q.correct_answer.split(";"):
        if "->" in part:
            k, v = part.split("->", 1)
            pairs[normalize_text(k)] = normalize_text(v)
        elif ":" in part:
            k, v = part.split(":", 1)
            pairs[normalize_text(k)] = normalize_text(v)

    total_pairs = len(q.left_items) if q.left_items else len(pairs)
    if total_pairs == 0:
        total_pairs = 1

    correct_count = 0
    for left_item, student_chosen in student_answer.items():
        left_norm = normalize_text(left_item)
        chosen_norm = normalize_text(student_chosen)
        if left_norm in pairs and pairs[left_norm] == chosen_norm:
            correct_count += 1

    score = int(round((correct_count / total_pairs) * marks))
    return GradedQuestion(
        question_index=q_idx,
        score=score,
        feedback=f"Matched {correct_count}/{total_pairs} pairs correctly. Awarded {score}/{marks} marks.",
        graded_by="deterministic"
    )

def grade_ordering(q: Question, student_answer: Any, q_idx: int) -> GradedQuestion:
    marks = q.marks
    if not student_answer or not isinstance(student_answer, list):
        return GradedQuestion(
            question_index=q_idx,
            score=0,
            feedback=f"No valid ordering provided. Correct order: {q.correct_answer}",
            graded_by="deterministic"
        )

    # Parse correct order
    # Format might be comma-separated or numbered: "1. A, 2. B"
    raw_parts = [p.strip() for p in q.correct_answer.split(",") if p.strip()]
    expected_order = []
    for p in raw_parts:
        cleaned = re.sub(r'^\d+[\.\)]\s*', '', p)
        expected_order.append(normalize_text(cleaned))

    total_items = len(expected_order)
    if total_items == 0:
        total_items = len(student_answer) or 1

    # Compare position-by-position
    correct_positions = 0
    for i, item in enumerate(student_answer):
        if i < len(expected_order) and normalize_text(item) == expected_order[i]:
            correct_positions += 1

    score = int(round((correct_positions / total_items) * marks))
    return GradedQuestion(
        question_index=q_idx,
        score=score,
        feedback=f"{correct_positions}/{total_items} items in the exact correct sequence. Awarded {score}/{marks} marks.",
        graded_by="deterministic"
    )

def grade_calculation(q: Question, student_answer: Any, q_idx: int) -> GradedQuestion:
    marks = q.marks
    ans_str = str(student_answer) if student_answer is not None else ""
    correct_str = str(q.correct_answer)

    # Extract first float/int from both
    num_pattern = r'[-+]?(?:\d*\.\d+|\d+)'
    ans_match = re.findall(num_pattern, ans_str)
    correct_match = re.findall(num_pattern, correct_str)

    if ans_match and correct_match:
        try:
            val_ans = float(ans_match[0])
            val_correct = float(correct_match[0])
            # Tolerance within 1% or 0.05
            tolerance = max(0.05, abs(val_correct * 0.01))
            if abs(val_ans - val_correct) <= tolerance:
                return GradedQuestion(
                    question_index=q_idx,
                    score=marks,
                    feedback=f"Correct calculation! Final answer {val_ans} matches expected value {val_correct}.",
                    graded_by="deterministic"
                )
        except Exception:
            pass

    return GradedQuestion(
        question_index=q_idx,
        score=0,
        feedback=f"Numerical answer did not match expected solution: {q.correct_answer}",
        graded_by="deterministic"
    )

OBJECTIVE_TYPES = {"mcq", "true_false", "matching", "ordering", "calculation"}

def is_objective_question(question_type: str) -> bool:
    return question_type.lower() in OBJECTIVE_TYPES

def grade_objective_question(q: Question, student_answer: Any, q_idx: int) -> Optional[GradedQuestion]:
    qtype = q.question_type.lower()
    if qtype == "mcq":
        return grade_mcq(q, student_answer, q_idx)
    elif qtype == "true_false":
        return grade_true_false(q, student_answer, q_idx)
    elif qtype == "matching":
        return grade_matching(q, student_answer, q_idx)
    elif qtype == "ordering":
        return grade_ordering(q, student_answer, q_idx)
    elif qtype == "calculation":
        return grade_calculation(q, student_answer, q_idx)
    return None
