from unittest.mock import patch, MagicMock
from client_adapter import is_backend_available, grade_exam_api

def test_is_backend_available_success():
    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        assert is_backend_available() is True

def test_is_backend_available_failure():
    with patch("requests.get", side_effect=Exception("Connection refused")):
        assert is_backend_available() is False

def test_grade_exam_api_request():
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {
        "graded_questions": [],
        "total_score": 10,
        "total_possible": 10
    }
    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = grade_exam_api(
            exam={"title": "Test Exam", "questions": []},
            student_answers={"0": "Ans"},
            api_key="mock-key",
            model_name="gemini-2.5-flash"
        )
        assert result["total_score"] == 10
        assert mock_post.called
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["headers"]["X-Gemini-API-Key"] == "mock-key"
        assert call_kwargs["json"]["model_name"] == "gemini-2.5-flash"
