from dataclasses import dataclass
from enum import Enum, auto


class ScoringDecision(Enum):
    """Возможные решения модели."""

    accepted = auto()
    declined = auto()


@dataclass
class ScoringResult(object):
    """Класс, содержащий результаты скоринга."""

    decision: ScoringDecision
    amount: int
    threshold: float
    proba: float


@dataclass
class Features(object):
    """Фичи для принятия решения об одобрении."""

    interest_rate: float = 0
    share_for_loan: float = 0
    weighted_ext: float = 0
    age: float = 0
    avg_income_per_adult: float = 0
    doc_change_years_ago: float = 0
    diff_days: float = 0
    application_credit_ratio: float = 0
    credit_goods_ratio: float = 0
    avg_children_per_adult: float = 0
    complete_home_info: float = 0
    doc_change_delay: float = 0
    amt_credit_limit_actual_full_mean: float = 0
    num_documents: float = 0
    share_active_debt: float = 0
