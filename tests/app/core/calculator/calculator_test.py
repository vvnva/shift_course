import pytest
from src.app.core.api import Features
from src.app.core.calculator import Calculator


class TestNewCalculator:
    """Тестирование нового калькулятора."""


    @pytest.fixture
    def calculator(self):
        return Calculator()


    @pytest.mark.parametrize(
        'proba, avg_income_per_adult, age, interest_rate, share_for_loan, weighted_ext, expected_amount',
        [
            (0.05, 10000, 30, 0.05, 0.2, 0.6, 425000),  # Минимальная вероятность дефолта, отсальные переменные ниже порога
            (0.1, 200000, 60, 0.08, 0.3, 0.7, 250000),  # Age category other amount
            (0.2, 150000, 56, 0.08, 0.3, 0.7, 250000),  # Age category other amount
            (0.05, 150000, 30, 0.08, 0.3, 0.7, 825000),
            (0.05, 10000, 30, 0.08, 0.3, 0.7, 525000),
            (0.05, 10000, 30, 0.05, 0.5, 0.7, 500000),
            (0.05, 10000, 30, 0.05, 0.2, 0.8, 465000),
        ]
    )


    def test_calc_amount(
            self,
            calculator,
            proba,
            avg_income_per_adult,
            age,
            interest_rate,
            share_for_loan,
            weighted_ext,
            expected_amount,
    ):
        features = Features(
            avg_income_per_adult=avg_income_per_adult,
            age=age,
            interest_rate=interest_rate,
            share_for_loan=share_for_loan,
            weighted_ext=weighted_ext,
        )
        assert calculator.calc_amount(proba, features) == expected_amount

