from Levenshtein import distance as lev

def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in s.strip() if ch.isalnum() or ch == " ")

def string_similarity(expected: str, predicted: str) -> float:
    a, b = _norm(expected), _norm(predicted)
    if not a and not b:
        return 1.0
    d = lev(a, b)  # Distância de Levenshtein
    return 1.0 - d / max(len(a), len(b))

def pronunciation_score(expected: str, predicted: str) -> dict:
    """
    Calcula a pontuação de pronúncia baseado na similaridade entre a palavra-alvo e o texto reconhecido.
    """
    sim = string_similarity(expected, predicted)  # 0..1
    hit = 1.0 if _norm(expected) == _norm(predicted) else 0.0  # Verifica se é uma correspondência exata
    score = 0.8 * sim + 0.2 * hit  # A pontuação final, ponderando a similaridade e o hit
    return {
        "score": round(100 * score, 1),
        "similarity": round(100 * sim, 1),
        "hit": bool(hit),
        "predicted": predicted,
    }
