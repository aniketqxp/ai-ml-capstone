"""
Phase 1 transcriber: PER-CHANNEL transcription (+ Phase 0 confidence capture).

Rationale
---------
The mono-mix approach blends both speakers, then *guesses* who spoke using RMS
energy. That guess is the dominant error source: ~200 correctly-transcribed words
land in the wrong speaker bucket on bad calls, each counted as a deletion+insertion.

The channels are physically separate microphone tracks (ch1 = agent, ch2 =
customer). So instead of mixing and guessing, we transcribe EACH channel
independently. The channel identity IS the speaker label -- attribution stops
being a heuristic and becomes a structural property of the data.

The original reason we avoided this (isolated channels are 53-84% silence ->
timestamp drift) is handled by faster-whisper's built-in Silero VAD
(vad_filter=True), which removes silence before transcription and remaps the
returned timestamps back onto the original recording timeline.

Phase 0 instrumentation
-----------------------
Every word carries its decoding confidence (`prob`) plus its parent segment's
`no_speech_prob` (ns) and `avg_logprob` (alp). These are captured but NOT used to
filter here -- storing them lets us experiment with confidence thresholds in
post-processing (e.g. dropping Whisper's silence-hallucinations) WITHOUT
re-transcribing. That is the whole point of instrumenting first.

The INITIAL_PROMPT is kept identical to run_batch.py so that "per-channel" is the
ONLY variable changing versus the mono-mix baseline -- a clean controlled A/B.
"""

import numpy as np
import soundfile as sf

SR = 16000

INITIAL_PROMPT = (
    "Banking and customer service call between an agent and a customer. "
    "Topics include accounts, transfers, payments, balances, and account numbers."
)


def _load_mono(path):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    return a, sr


def transcribe_channel(model, audio, speaker, decode=None):
    """
    Transcribe one isolated channel. Returns (words, segments_meta).
    words:   list of {word, start, end, prob, ns, alp}
    segments_meta: list of {start, end, text, avg_logprob, no_speech_prob,
                            compression_ratio, n_words} -- the Phase 0 record.

    decode: optional dict of overrides passed to model.transcribe(). Keys like
            vad_filter, beam_size, condition_on_previous_text, no_speech_threshold,
            etc. A nested "vad_parameters" dict is MERGED onto the defaults so you
            can tweak just threshold/min_silence without restating the whole dict.
            Defaults reproduce the original Phase-1 config exactly.
    """
    opts = dict(
        language="en",
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        initial_prompt=INITIAL_PROMPT,
    )
    if decode:
        decode = dict(decode)
        vp = decode.pop("vad_parameters", None)
        if vp:
            opts["vad_parameters"] = {**(opts.get("vad_parameters") or {}), **vp}
        opts.update(decode)
        # if VAD is disabled, drop vad_parameters so faster-whisper doesn't warn
        if opts.get("vad_filter") is False:
            opts.pop("vad_parameters", None)
    seg_gen, _info = model.transcribe(audio, **opts)

    words, segments_meta = [], []
    for seg in seg_gen:
        ns  = round(float(seg.no_speech_prob), 4)
        alp = round(float(seg.avg_logprob), 4)
        n_seg_words = 0
        if seg.words:
            for w in seg.words:
                if w.start is None or w.end is None:
                    continue
                words.append({
                    "word":  w.word.strip(),
                    "start": round(w.start, 3),
                    "end":   round(w.end, 3),
                    "prob":  round(float(w.probability), 4),
                    "ns":    ns,
                    "alp":   alp,
                })
                n_seg_words += 1
        segments_meta.append({
            "start": round(seg.start, 3),
            "end":   round(seg.end, 3),
            "text":  seg.text.strip(),
            "avg_logprob":       alp,
            "no_speech_prob":    ns,
            "compression_ratio": round(float(seg.compression_ratio), 4),
            "n_words": n_seg_words,
        })
    return words, segments_meta


def transcribe_call(model, agent_path, customer_path, preprocess_fn=None, decode=None):
    """
    Per-channel transcription of one call.
    preprocess_fn: optional callable(audio)->audio applied to each channel before
                   transcription (Phase 2: high-pass + loudness normalization).
    decode: optional dict of model.transcribe() overrides (see transcribe_channel).
    Returns dict pieces: (agent_words, customer_words, segments, duration_s).
    """
    a_audio, a_sr = _load_mono(agent_path)
    c_audio, c_sr = _load_mono(customer_path)

    if preprocess_fn is not None:
        a_audio = preprocess_fn(a_audio)
        c_audio = preprocess_fn(c_audio)

    agent_words,    agent_segs    = transcribe_channel(model, a_audio, "agent", decode)
    customer_words, customer_segs = transcribe_channel(model, c_audio, "customer", decode)

    duration = max(len(a_audio) / SR, len(c_audio) / SR)
    segments = {"agent": agent_segs, "customer": customer_segs}
    return agent_words, customer_words, segments, duration
