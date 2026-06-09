"""
Shared evaluation utilities — single source of truth for text normalisation
so the WER scorer and the error analyser never drift apart.

Uses the Whisper EnglishTextNormalizer (the published OpenAI normaliser) as the
canonical text standard. It handles, in a principled and citeable way:
  - number words <-> digits   ("four hundred" == "400")
  - contraction expansion     ("you'd" == "you would")
  - British/American spelling
  - filler/interjection removal
  - punctuation & casing

We add a light pre-pass to strip the dataset's disfluency annotations
( (uh), (um), word~ stutter markers ) which are transcription conventions,
not spoken words.
"""

import re
from whisper_normalizer.english import EnglishTextNormalizer

_whisper_norm = EnglishTextNormalizer()


def _strip_dataset_annotations(text):
    # Filled pauses in parentheses: (uh), (um), (ah), (hm), (mm), (er), (mhm)...
    text = re.sub(r'\(\s*[aehmu]+[mh]?\s*[,\.]?\s*\)', ' ', text, flags=re.IGNORECASE)
    # Any remaining short parenthetical disfluency markers
    text = re.sub(r'\([^)]{1,15}\)', ' ', text)
    # Stutter markers:  word~  or  w~
    text = re.sub(r'\w+~\s*', '', text)
    return text


def normalise(text):
    """Canonical normalisation applied identically to reference and hypothesis."""
    text = _strip_dataset_annotations(text)
    text = _whisper_norm(text)          # the heavy lifting: numbers, contractions, etc.
    return text.strip()


def normalise_raw(text):
    """
    Conservative normalisation: lowercase + strip punctuation only.
    Does NOT canonicalise numbers or expand contractions.
    Used to report the honest, pessimistic 'raw WER' alongside the
    semantic 'normalised WER'.
    """
    text = _strip_dataset_annotations(text)
    text = text.replace('’', "'").replace('‘', "'")
    text = re.sub(r"[^\w\s']", ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


if __name__ == "__main__":
    # Quick self-test
    tests = [
        ("four hundred dollars", "400 dollars"),
        ("you'd say it's the same", "you would say it is the same"),
        ("on the 2nd of November", "on the second of november"),
        ("(uh) my name is James~ James Carter", "my name is james carter"),
    ]
    print("Whisper normaliser self-test:")
    for a, b in tests:
        na, nb = normalise(a), normalise(b)
        match = "MATCH" if na == nb else "DIFFER"
        print(f"  [{match}] '{a}' -> '{na}'")
        print(f"          '{b}' -> '{nb}'")
