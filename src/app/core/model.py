from src.app.core.api import Features, ScoringDecision, ScoringResult
from src.app.core.calculator import Calculator
import joblib


class Model_with_scoring(object):
    """Класс для моделей c расчетом proba и threshold."""

    _threshold = 0.25

    def __init__(self, model_path: str):
        """Создает объект класса."""
        self._model = joblib.load(model_path)
        self._Calculator = Calculator()

    def get_scoring_result(self, features: Features) -> ScoringResult:
        """Возвращает объект ScoringResult с результатами скоринга."""
        proba = self._predict_proba(features)

        decision = ScoringDecision.DECLINED
        amount = 0

        if proba < self._threshold:
            decision = ScoringDecision.ACCEPTED
            amount = self._Calculator.calc_amount(proba, features)

        return ScoringResult(
            decision=decision,
            amount=amount,
            threshold=self._threshold,
            proba=proba,
        )

    def _predict_proba(self, features: Features) -> float:
        """Определяет вероятность невозврата займа."""
        return self._model.predict_proba([list(features.__dict__.values())])[0, 1]
