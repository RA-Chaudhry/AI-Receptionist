"""
tests/test_vocabulary_correction.py — Post-STT Vocabulary Correction Tests
"""

import pytest
from app.core.config import settings
from app.utils.entity_normalizer import correct_voice_vocabulary


def test_vocabulary_correction_misspelled_variants():
    """Verify misheard company/brand variants are mapped to canonical 'Logic Pulse'."""
    cases = [
        ("logicals", "Logic Pulse"),
        ("Logicals", "Logic Pulse"),
        ("logical", "Logic Pulse"),
        ("Logical", "Logic Pulse"),
        ("ligit pulse", "Logic Pulse"),
        ("Ligit Pulse", "Logic Pulse"),
        ("logic pals", "Logic Pulse"),
        ("Logic Pals", "Logic Pulse"),
        ("logic pulls", "Logic Pulse"),
        ("lodge pulse", "Logic Pulse"),
        ("Lodge Pulse", "Logic Pulse"),
    ]
    for raw, expected in cases:
        assert correct_voice_vocabulary(raw) == expected, f"Failed correcting '{raw}'"


def test_vocabulary_correction_in_conversational_queries():
    """Verify misheard terms inside user sentences are replaced accurately."""
    queries = [
        (
            "What services does logical offer?",
            "What services does Logic Pulse offer?",
        ),
        (
            "Tell me about ligit pulse timings.",
            "Tell me about Logic Pulse timings.",
        ),
        (
            "Who founded logic pals?",
            "Who founded Logic Pulse?",
        ),
        (
            "Where is lodge pulse office located?",
            "Where is Logic Pulse office located?",
        ),
        (
            "I want to book an appointment with logicals please.",
            "I want to book an appointment with Logic Pulse please.",
        ),
    ]
    for raw, expected in queries:
        assert correct_voice_vocabulary(raw) == expected


def test_vocabulary_correction_normal_sentences_not_mangled():
    """Verify standard English/Urdu sentences and partial substrings are preserved."""
    normal_sentences = [
        "This is arranged in chronological order.",
        "That is an illogical conclusion to make.",
        "Can I schedule an appointment for Friday at 3 PM?",
        "Do you build web applications and mobile dashboards?",
        "What is the office timing of Muhammad Ramzan?",
        "Main software development ke baare mein maloomat chahta hoon.",
        "Mera phone number 03001234567 hai.",
    ]
    for sentence in normal_sentences:
        assert correct_voice_vocabulary(sentence) == sentence, f"Mangled normal sentence: '{sentence}'"


def test_vocabulary_correction_custom_config(monkeypatch):
    """Verify vocabulary_corrections and vocabulary_target can be customized via config."""
    monkeypatch.setattr(settings, "vocabulary_corrections", ["foo bot", "foobot"])
    monkeypatch.setattr(settings, "vocabulary_target", "BarBot")

    assert correct_voice_vocabulary("Hello foobot!") == "Hello BarBot!"
    assert correct_voice_vocabulary("Talk to foo bot today.") == "Talk to BarBot today."
