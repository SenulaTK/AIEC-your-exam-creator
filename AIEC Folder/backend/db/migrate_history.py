"""
Migration script to parse existing `aiec_exam_history.json` and generate
PostgreSQL INSERT statements or directly migrate into Cloud SQL.
"""
import json
import uuid
from pathlib import Path
from typing import Optional

def generate_migration_sql(json_path: Path, output_sql_path: Optional[Path] = None) -> str:
    if not json_path.exists():
        raise FileNotFoundError(f"Source history file not found: {json_path}")

    history = json.loads(json_path.read_text())
    print(f"Loaded {len(history)} exam entries from {json_path}")

    statements = []
    statements.append("-- Generated AIEC Migration Script --\n")
    statements.append("BEGIN;\n")

    # Ensure a default migration user exists
    default_user_id = "00000000-0000-0000-0000-000000000001"
    statements.append(
        f"INSERT INTO users (id, email, role) VALUES ('{default_user_id}', 'migrated_teacher@aiec.local', 'teacher') "
        "ON CONFLICT (email) DO NOTHING;\n"
    )

    for item in history:
        exam_id = item.get("id") or str(uuid.uuid4())
        created_at = item.get("timestamp", "2026-01-01 00:00:00")
        exam_data = item.get("exam", {})
        title = (exam_data.get("title") or item.get("title", "Untitled Exam")).replace("'", "''")
        instructions = (exam_data.get("instructions") or "").replace("'", "''")

        # Insert exam
        statements.append(
            f"INSERT INTO exams (id, teacher_id, title, instructions, created_at) "
            f"VALUES ('{exam_id}', '{default_user_id}', '{title}', '{instructions}', '{created_at}') "
            f"ON CONFLICT (id) DO NOTHING;"
        )

        # Insert questions
        questions = exam_data.get("questions", [])
        for idx, q in enumerate(questions):
            q_id = str(uuid.uuid4())
            q_text = (q.get("question_text") or "").replace("'", "''")
            q_type = q.get("question_type", "written")
            topic = (q.get("topic") or "").replace("'", "''")
            difficulty = q.get("difficulty", "medium")
            marks = int(q.get("marks", 1))
            correct_ans = (q.get("correct_answer") or "").replace("'", "''")
            
            options_json = json.dumps(q.get("options")).replace("'", "''") if q.get("options") else "NULL"
            if options_json != "NULL":
                options_json = f"'{options_json}'::jsonb"

            metadata = {
                "left_items": q.get("left_items"),
                "right_items": q.get("right_items"),
                "items": q.get("items"),
                "categories": q.get("categories"),
                "label_prompts": q.get("label_prompts"),
                "expected_units": q.get("expected_units"),
            }
            meta_json = json.dumps(metadata).replace("'", "''")

            statements.append(
                f"INSERT INTO questions (id, exam_id, question_index, question_text, question_type, topic, difficulty, marks, options, correct_answer, metadata) "
                f"VALUES ('{q_id}', '{exam_id}', {idx}, '{q_text}', '{q_type}', '{topic}', '{difficulty}', {marks}, {options_json}, '{correct_ans}', '{meta_json}'::jsonb);"
            )

        # Insert submission results if present
        results = item.get("results")
        if results and isinstance(results, dict):
            sub_id = str(uuid.uuid4())
            graded_qs = results.get("graded_questions", [])
            total_score = sum(g.get("score", 0) for g in graded_qs)
            total_possible = sum(q.get("marks", 0) for q in questions)
            
            statements.append(
                f"INSERT INTO submissions (id, exam_id, student_name, total_score, total_possible, submitted_at) "
                f"VALUES ('{sub_id}', '{exam_id}', 'Historical Student', {total_score}, {total_possible}, '{created_at}');"
            )

            for g in graded_qs:
                ans_id = str(uuid.uuid4())
                q_idx = g.get("question_index", 0)
                score = g.get("score", 0)
                feedback = (g.get("feedback") or "").replace("'", "''")
                statements.append(
                    f"INSERT INTO submission_answers (id, submission_id, question_index, score, feedback) "
                    f"VALUES ('{ans_id}', '{sub_id}', {q_idx}, {score}, '{feedback}');"
                )

        statements.append("\n")

    statements.append("COMMIT;\n")
    full_sql = "\n".join(statements)

    if output_sql_path:
        output_sql_path.write_text(full_sql)
        print(f"Migration SQL successfully written to: {output_sql_path}")

    return full_sql

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    src_json = base_dir / "aiec_exam_history.json"
    dest_sql = Path(__file__).parent / "migrated_history.sql"
    generate_migration_sql(src_json, dest_sql)
