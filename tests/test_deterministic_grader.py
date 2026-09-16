from backend.app.models.exam import Question
from backend.app.services.deterministic_grader import (
    grade_mcq,
    grade_true_false,
    grade_matching,
    grade_ordering,
    grade_calculation,
    grade_objective_question,
    is_objective_question
)

def test_is_objective_question():
    assert is_objective_question("mcq") is True
    assert is_objective_question("true_false") is True
    assert is_objective_question("matching") is True
    assert is_objective_question("ordering") is True
    assert is_objective_question("calculation") is True
    assert is_objective_question("essay") is False
    assert is_objective_question("short_answer") is False

def test_grade_mcq_exact_match():
    q = Question(
        question_text="What is the powerhouse of the cell?",
        question_type="mcq",
        marks=2,
        options=["Nucleus", "Ribosome", "Mitochondria", "Chloroplast"],
        correct_answer="Mitochondria"
    )
    # Exact text
    res = grade_mcq(q, "Mitochondria", 0)
    assert res.score == 2
    assert "Correct!" in res.feedback

    # Case insensitive
    res = grade_mcq(q, "mitochondria", 0)
    assert res.score == 2

    # Wrong answer
    res_wrong = grade_mcq(q, "Nucleus", 0)
    assert res_wrong.score == 0
    assert "Incorrect" in res_wrong.feedback

def test_grade_mcq_letter_option():
    q = Question(
        question_text="Pick the primary sector activity.",
        question_type="mcq",
        marks=1,
        options=["Engineering", "Mining", "Retailing"],
        correct_answer="Mining"
    )
    # Option index 1 (second option -> 'b')
    res = grade_mcq(q, "b", 0)
    assert res.score == 1

def test_grade_true_false():
    q_true = Question(
        question_text="Water boils at 100C at sea level.",
        question_type="true_false",
        marks=1,
        correct_answer="True"
    )
    assert grade_true_false(q_true, "True", 0).score == 1
    assert grade_true_false(q_true, "true", 0).score == 1
    assert grade_true_false(q_true, "False", 0).score == 0

    q_false = Question(
        question_text="Indemnity allows you to make a profit.",
        question_type="true_false",
        marks=1,
        correct_answer="False. Indemnity restores original position."
    )
    assert grade_true_false(q_false, "False", 0).score == 1
    assert grade_true_false(q_false, "True", 0).score == 0

def test_grade_matching():
    q = Question(
        question_text="Match the terms",
        question_type="matching",
        marks=4,
        left_items=["Alliteration", "Hyperbole", "Oxymoron", "Onomatopoeia"],
        correct_answer="Alliteration -> Peter Piper; Hyperbole -> Million times; Oxymoron -> Deafening silence; Onomatopoeia -> Sizzled"
    )
    student_ans = {
        "Alliteration": "Peter Piper",
        "Hyperbole": "Million times",
        "Oxymoron": "Deafening silence",
        "Onomatopoeia": "Sizzled"
    }
    res = grade_matching(q, student_ans, 0)
    assert res.score == 4

    # Half correct
    student_half = {
        "Alliteration": "Peter Piper",
        "Hyperbole": "Million times",
        "Oxymoron": "Wrong",
        "Onomatopoeia": "Wrong"
    }
    res_half = grade_matching(q, student_half, 0)
    assert res_half.score == 2

def test_grade_ordering():
    q = Question(
        question_text="Order stages",
        question_type="ordering",
        marks=4,
        items=["Raw materials", "Manufacturing", "Wholesale", "Retail"],
        correct_answer="1. Raw materials, 2. Manufacturing, 3. Wholesale, 4. Retail"
    )
    # Perfect order
    res = grade_ordering(q, ["Raw materials", "Manufacturing", "Wholesale", "Retail"], 0)
    assert res.score == 4

    # Partial order (2 out of 4 in place)
    res_partial = grade_ordering(q, ["Raw materials", "Manufacturing", "Retail", "Wholesale"], 0)
    assert res_partial.score == 2

def test_grade_calculation():
    q = Question(
        question_text="Calculate 1399 / 12",
        question_type="calculation",
        marks=2,
        expected_units="ZAR",
        correct_answer="116.58 ZAR"
    )
    # Exact
    res = grade_calculation(q, "116.58", 0)
    assert res.score == 2

    # With units
    res_unit = grade_calculation(q, "116.58 ZAR", 0)
    assert res_unit.score == 2

    # Within slight floating rounding
    res_float = grade_calculation(q, 116.583, 0)
    assert res_float.score == 2

    # Wrong
    res_wrong = grade_calculation(q, "50.00", 0)
    assert res_wrong.score == 0

def test_grade_objective_question_dispatcher():
    q = Question(
        question_text="2 + 2 = ?",
        question_type="mcq",
        marks=1,
        options=["3", "4", "5"],
        correct_answer="4"
    )
    res = grade_objective_question(q, "4", 0)
    assert res is not None
    assert res.score == 1

    non_obj = Question(
        question_text="Explain osmosis",
        question_type="essay",
        marks=5,
        correct_answer="Rubric"
    )
    assert grade_objective_question(non_obj, "Essay text", 1) is None

