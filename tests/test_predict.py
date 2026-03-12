from src.inference.predict import IntentPredictor


def test_predict_returns_expected_keys():
    predictor = IntentPredictor()
    result = predictor.predict("Please help me access my account.")

    assert "text" in result
    assert "intent" in result
    assert "confidence" in result
    assert "probabilities" in result


def test_predict_intent_is_string():
    predictor = IntentPredictor()
    result = predictor.predict("Thank you for the amazing support.")

    assert isinstance(result["intent"], str)


def test_predict_confidence_is_float():
    predictor = IntentPredictor()
    result = predictor.predict("URGENT: the dashboard is down.")

    assert isinstance(result["confidence"], float)
    