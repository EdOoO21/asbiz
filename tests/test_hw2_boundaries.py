"""Границы регламента и сохранение проверок исходного обращения."""
import asyncio
import json

import pytest
from pydantic import ValidationError

from desk.escalation import Draft, Signals, atriage_many, needs_human
from tests.fakes import FakeLLM

NONE = dict(refund=None, duplicate=None, fraud=False, threat=False,
            asks_human=False, tariff_pro=False, merchant_down=False, key_leak=False)
SOURCE = 'За заказ P-61234 и P-61235 списали 5000 рублей.'
GOOD = dict(reasoning='Вопрос о списании.', category='платежи', severity=2,
            quote='списали 5000 рублей', payment_ids=['P-61234', 'P-61235'],
            amount=5000, signals=NONE)


@pytest.mark.parametrize('field,amount,expected', [
    ('refund', None, False), ('refund', 0, False), ('refund', 5000, False),
    ('refund', 5001, True), ('duplicate', None, False), ('duplicate', 0, False),
    ('duplicate', 15000, False), ('duplicate', 15001, True),
])
def test_strict_money_thresholds(field, amount, expected):
    assert needs_human(Signals(**{**NONE, field: amount})) is expected


@pytest.mark.parametrize('field', ['fraud', 'threat', 'asks_human', 'tariff_pro', 'merchant_down', 'key_leak'])
def test_each_boolean_signal_is_sufficient(field):
    assert needs_human(Signals(**{**NONE, field: True})) is True


@pytest.mark.parametrize('changes', [
    {'payment_ids': []}, {'payment_ids': ['P-61234']},
    {'payment_ids': ['P-99999']}, {'payment_ids': ['61234']},
    {'quote': 'выдуманная цитата'}, {'quote': ''},
    {'severity': 0}, {'severity': 6}, {'category': 'неизвестно'}, {'amount': -1},
])
def test_draft_rejects_invalid_extraction(changes):
    with pytest.raises(ValidationError):
        Draft.model_validate({**GOOD, **changes}, context={'source': SOURCE})


def test_default_ids_are_checked_against_source():
    answer = {k:v for k,v in GOOD.items() if k != 'payment_ids'}
    with pytest.raises(ValidationError):
        Draft.model_validate(answer, context={'source': SOURCE})


def test_batch_preserves_error_position_and_ignores_model_decision():
    answer = {**GOOD, 'needs_human': True}
    llm = FakeLLM(['broken'] * 3 + [json.dumps(answer)])
    results = asyncio.run(atriage_many(llm, [SOURCE, SOURCE], concurrency=1))
    assert isinstance(results[0], Exception)
    assert results[1].needs_human is False
    assert results[1].payment_ids == GOOD['payment_ids']
    assert 'needs_human' not in Draft.model_fields


def test_empty_batch_makes_no_calls():
    llm = FakeLLM([])
    assert asyncio.run(atriage_many(llm, [])) == []
    assert llm.calls == []
