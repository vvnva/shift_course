from src.app.core.api import Features


class Calculator(object):
    """Класс для расчета одобренной суммы."""

    proba_threshold = 0.1
    income_threshold = 140000

    age_category_other_amount = 150000
    default_amount = 25000

    interest_rate_threshold = 0.07
    share_for_loan_threshold = 0.23
    weighted_ext_threshold = 0.65
    age_threshold = 55

    adds = [400000, 300000, 25000, 35000, 40000]

    def calc_amount(
        self,
        proba: float,
        features: Features,
    ) -> int:
        """Функция принимает на вход вероятность дефолта и признаки и расчитывает одобренную сумму."""
        # варирование по age, interest_rate_threshold, weighted_ext_threshold скору, share for loan, avg_income_per_adult

        amount = self.default_amount
        if features.age > self.age_threshold:
            amount = self.age_category_other_amount
        if proba < self.proba_threshold:
            amount += self.adds[0]
        if features.avg_income_per_adult > self.income_threshold:
            amount += self.adds[1]
        if features.interest_rate > self.interest_rate_threshold:
            amount += self.adds[2]
        if features.share_for_loan > self.share_for_loan_threshold:
            amount += self.adds[3]
        if features.weighted_ext > self.weighted_ext_threshold:
            amount += self.adds[4]

        return amount
