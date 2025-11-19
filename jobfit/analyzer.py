"""Analizador simple para ofertas y CVs.

Heurísticas:
- Extrae texto de una URL (HTML -> texto) o usa texto dado.
- Busca secciones con palabras clave (Requirements, Requisitos, Skills, Responsibilities)
- Parseo simple de CV por líneas y búsqueda de frases que contengan skills.
- No añade información nueva; la adaptación del CV solo reordena y resalta evidencias existentes.
"""
from typing import List, Dict, Tuple, Optional
import re
import requests
from bs4 import BeautifulSoup
import numpy as np

# Optional imports for embeddings/FAISS
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except Exception:
    SentenceTransformer = None  # type: ignore
    faiss = None  # type: ignore
try:
    from pdfminer.high_level import extract_text as extract_text_from_pdfminer
except Exception:
    extract_text_from_pdfminer = None  # type: ignore

KEYWORDS_REQUIREMENTS = [
    "requirements",
    "requisitos",
    "responsibilities",
    "responsabilidades",
    "skills",
    "habilidades",
    "qualifications",
    "calificaciones",
]


def fetch_job_posting_from_url(url: str) -> str:
    """Descarga HTML y devuelve el texto visible.

    No sigue JS; funciona para páginas estáticas.
    """
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # eliminar scripts y estilos
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    text = soup.get_text(separator="\n")
    # normalizar espacios
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def extract_requirements(text: str) -> List[str]:
    """Extrae bloques de texto que parezcan requisitos.

    Devuelve lista de frases cortas (una por requisito).
    """
    lower = text.lower()
    # Buscar palabras clave y tomar secciones alrededor
    candidates = []
    for kw in KEYWORDS_REQUIREMENTS:
        idx = lower.find(kw)
        if idx != -1:
            start = max(0, idx - 200)
            end = min(len(text), idx + 2000)
            candidates.append(text[start:end])

    if not candidates:
        # fallback: tomar los párrafos más largos como candidato
        parts = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
        candidates = parts[:3]

    requirements = []
    for block in candidates:
        # dividir por líneas y por viñetas
        lines = re.split(r"\n|\r|\u2022|\-|\*", block)
        for line in lines:
            s = line.strip()
            if len(s) < 10:
                continue
            # filtrar líneas demasiado genéricas
            if len(s.split()) > 2:
                requirements.append(s)

    # deduplicar manteniendo orden
    seen = set()
    reqs = []
    for r in requirements:
        key = r.lower()
        if key not in seen:
            seen.add(key)
            reqs.append(r)
    return reqs


def build_embeddings_index(sentences: List[str], model_name: str = "all-MiniLM-L6-v2") -> Tuple[Optional[object], Optional[object], List[str]]:
    """Build a FAISS index for the provided sentences using sentence-transformers.

    Returns (model, index, sentences). If dependencies not installed returns (None, None, sentences).
    """
    if SentenceTransformer is None or faiss is None:
        return None, None, sentences

    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
    if len(embeddings.shape) == 1:
        embeddings = np.expand_dims(embeddings, 0)
    embeddings = embeddings.astype('float32')
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    # normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return model, index, sentences


def semantic_search(query: str, model: object, index: object, sentences: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
    """Return top_k sentences similar to query using provided model and FAISS index.

    Returns list of (sentence, score) ordered by score desc.
    If model/index are None, returns empty list.
    """
    if model is None or index is None:
        return []
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb.astype('float32')
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results: List[Tuple[str, float]] = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0 or idx >= len(sentences):
            continue
        results.append((sentences[idx], float(score)))
    return results


def sentence_tokenize(text: str) -> List[str]:
    # muy simple: separar por puntos y nuevas líneas
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def parse_cv_text(text: str) -> Dict[str, List[str]]:
    """Parseo heurístico del CV en texto.

    Devuelve dict con claves: 'skills', 'experience', 'education', 'all_sentences'
    """
    sentences = sentence_tokenize(text)
    skills = []
    experience = []
    education = []

    # heurística: líneas con 'skills', 'tecnologías', 'lenguajes' => recoger palabras
    lower = text.lower()
    if "skills" in lower or "habilidades" in lower or "tecnolog" in lower:
        # buscar una línea con esas palabras
        for line in text.splitlines():
            l = line.strip()
            if re.search(r"skills|habilidades|tecnolog|lenguajes|tools", l, re.I):
                # extraer tokens separados por coma
                parts = re.split(r",|;|\||/", l)
                for p in parts:
                    token = p.strip()
                    if len(token) > 1 and len(token.split()) < 6:
                        skills.append(token)

    # experiencia: buscar secciones con años, cargos, empresas
    for s in sentences:
        if re.search(r"\b(\d{4}|\bexperience\b|experiencia|responsible|responsabilidades)\b", s, re.I):
            experience.append(s)
        if re.search(r"education|estudios|grado|licenciatur|master|universi", s, re.I):
            education.append(s)

    # fallback: si skills vacío, extraer nouns cortos frecuentes
    if not skills:
        tokens = re.findall(r"[A-Za-z#+\-\.]+", text)
        freq = {}
        for t in tokens:
            tlow = t.lower()
            if len(t) <= 2:
                continue
            freq[tlow] = freq.get(tlow, 0) + 1
        # elegir tokens más frecuentes
        popular = sorted(freq.items(), key=lambda x: -x[1])[:30]
        skills = [p[0] for p in popular]

    return {
        "skills": skills,
        "experience": experience,
        "education": education,
        "all_sentences": sentences,
    }


def extract_text_from_pdf(path: str) -> str:
    """Extract text from PDF using pdfminer if available, otherwise return empty string."""
    if extract_text_from_pdfminer is None:
        return ""
    try:
        return extract_text_from_pdfminer(path)
    except Exception:
        return ""


def estimate_realism_of_job_posting(text: str) -> float:
    """Heurística para estimar un 'score de realismo' en [0,1].

    Señales consideradas:
    - presencia de nombre de empresa (máculas/Word capitalized)
    - presencia de contacto (email o URL)
    - mención de salario o rango
    - existencia de secciones 'Requirements'/'Responsabilities' etc.
    - longitud mínima del anuncio
    - penalización por lenguaje exagerado (buzzwords: 'rockstar', 'ninja')
    """
    score = 0.0
    text_lower = text.lower()

    # company-like indicator: presence of 'company' or some capitalized word preceding 'Inc'/'LLC' or 'S.A.'
    if re.search(r"\b(company|inc|llc|s\.a\.|corp|ltd)\b", text, re.I):
        score += 0.2

    # contact: email or http
    if re.search(r"[\w\.\-]+@[\w\.\-]+\.[a-z]{2,}", text_lower) or re.search(r"https?://", text_lower):
        score += 0.2

    # salary
    if re.search(r"\b(\$|€|£)\s?\d{2,}|\b(salary|compensation|per year|per annum|annual)\b", text_lower):
        score += 0.15

    # sections
    if any(k in text_lower for k in ("requirements", "responsibilities", "responsabilidades", "skills", "habilidades")):
        score += 0.2

    # length
    words = len(re.findall(r"\w+", text))
    if words >= 80:
        score += 0.1

    # penalizar buzzwords
    buzz = re.findall(r"\b(rockstar|ninja|guru|superstar|unlimited)\b", text_lower)
    if buzz:
        score -= 0.2

    # normalizar
    final = max(0.0, min(1.0, score))
    return round(final, 2)


def audit_requirements(requirements: List[str], cv_parsed: Dict[str, List[str]], *, use_embeddings: bool = False, model: Optional[object] = None, index: Optional[object] = None, sentences: Optional[List[str]] = None) -> List[Dict]:
    """Para cada requisito, devuelve si hay evidencia y la(s) oracion(es) que la prueban.

    No inventa; sólo busca coincidencias (palabras clave o frases) en el CV.
    """
    results = []
    sentences_all = cv_parsed.get("all_sentences", [])
    for req in requirements:
        evidence: List[str] = []
        skills_hits: List[str] = []
        score = 0.0

        if use_embeddings and model is not None and index is not None and sentences is not None:
            # Semántic retrieval
            hits = semantic_search(req, model, index, sentences, top_k=5)
            for sent, sc in hits:
                evidence.append(sent)
            # derive score from best hit
            if hits:
                best_score = hits[0][1]
                # normalize IP cosine score [-1,1] to [0,1]
                norm = max(0.0, min(1.0, (best_score + 1.0) / 2.0))
                score = round(norm, 2)
            # skills hits: check skills list for tokens in req
            for sk in cv_parsed.get("skills", []):
                skl = sk.lower()
                if skl in req.lower() or any(tok in skl for tok in re.findall(r"[A-Za-z#+\-\.]+", req.lower()) if len(tok) > 2):
                    skills_hits.append(sk)
        else:
            # fallback to simple heuristic matching
            req_lower = req.lower()
            tokens = [t for t in re.findall(r"[A-Za-z#+\-\.]+", req_lower) if len(t) > 2]
            for s in sentences_all:
                sl = s.lower()
                match_count = sum(1 for t in tokens if t in sl)
                if match_count >= max(1, len(tokens) // 4):
                    evidence.append(s)
            for sk in cv_parsed.get("skills", []):
                skl = sk.lower()
                for t in tokens:
                    if t in skl or skl in req_lower:
                        skills_hits.append(sk)
                        break
            if evidence:
                score = min(1.0, 0.4 + 0.15 * len(evidence))
            if skills_hits:
                score = max(score, min(1.0, 0.3 + 0.1 * len(skills_hits)))

        # mejorar scoring combinando señales:
        # - si tenemos embedding: usar similitud semántica (score ya contiene normalizado)
        # - añadir bono por matches explícitos en skills
        # - añadir bono por coincidencias exactas de tokens
        bonus = 0.0
        if skills_hits:
            bonus += min(0.25, 0.05 * len(skills_hits))
        # exact token matches
        tokens_req = [t for t in re.findall(r"[A-Za-z#+\\-\\.]+", req.lower()) if len(t) > 2]
        exact_matches = 0
        for t in tokens_req:
            for s in sentences_all:
                if re.search(r"\\b" + re.escape(t) + r"\\b", s.lower()):
                    exact_matches += 1
                    break
        if exact_matches:
            bonus += min(0.25, 0.05 * exact_matches)

        final_score = min(1.0, score + bonus)

        results.append({
            "requirement": req,
            "score": round(final_score, 2),
            "raw_score": round(score, 2),
            "bonus": round(bonus, 2),
            "evidence_sentences": evidence,
            "skills_matches": skills_hits,
        })
    return results


def generate_adapted_cv_from_selected_evidence(original_text: str, selected_evidence: List[str]) -> str:
    """Genera una versión adaptada del CV resaltando SOLO las evidencias seleccionadas.

    Esta función no inventa nada: únicamente resalta las oraciones seleccionadas en el texto original.
    """
    adapted = original_text
    # evitar reemplazos múltiples idénticos repetidos: deduplicar y ordenar por longitud (largo primero)
    unique = []
    seen = set()
    for e in selected_evidence:
        if e not in seen and e.strip():
            seen.add(e)
            unique.append(e)
    unique = sorted(unique, key=lambda s: -len(s))
    for sent in unique:
        if sent in adapted:
            adapted = adapted.replace(sent, f"[MATCH] {sent} [MATCH]")
    return adapted


def adapt_cv(requirements: List[str], cv_parsed: Dict[str, List[str]], original_text: str) -> str:
    """Genera una versión adaptada del CV en texto resaltando evidencia existente.

    No añade logros, fechas o roles que no estén presentes en el CV original.
    Resalta frases con el prefijo [MATCH] para que el usuario las revise.
    """
    audit = audit_requirements(requirements, cv_parsed)
    highlighted = original_text
    # Para cada evidencia, envolver la oración en [MATCH] ... [/MATCH]
    for item in audit:
        for sent in item.get("evidence_sentences", []):
            # escapar paréntesis y crear replace seguro
            if sent in highlighted:
                highlighted = highlighted.replace(sent, f"[MATCH] {sent} [MATCH]")

    # Recomendación: poner skills importantes al inicio si aparecen
    intro_lines = []
    skills = cv_parsed.get("skills", [])
    if skills:
        intro_lines.append("Skills destacadas: " + ", ".join(skills[:10]))

    adapted = "\n\n".join(intro_lines + [highlighted]) if intro_lines else highlighted
    return adapted


def justify_fit(requirements: List[str], audit_results: List[Dict]) -> List[str]:
    """Crea explicaciones concisas para cada requisito con evidencias del CV.

    Devuelve lista de cadenas: 'Requisito: Justificación (evidencia)'.
    """
    lines = []
    for r in audit_results:
        req = r["requirement"]
        score = r["score"]
        evid = r.get("evidence_sentences", [])
        skills = r.get("skills_matches", [])
        parts = []
        if evid:
            parts.append("Evidencia: " + "; ".join(evid[:2]))
        if skills:
            parts.append("Skills relacionadas: " + ", ".join(skills[:5]))
        if not parts:
            parts.append("No se encontró evidencia directa en el CV.")
        lines.append(f"{req} — Encaje: {score}. {' | '.join(parts)}")
    return lines


if __name__ == "__main__":
    # demo rápido si se ejecuta solo
    sample_job = "We are looking for a Python developer with experience in Django, REST APIs and PostgreSQL. Strong skills: Python, Django, SQL."
    sample_cv = "Juan Pérez\nSkills: Python, Django, Flask, SQL\nExperience: 3 years building REST APIs with Django and Postgres."
    reqs = extract_requirements(sample_job)
    parsed = parse_cv_text(sample_cv)
    audit = audit_requirements(reqs, parsed)
    print('\n'.join(justify_fit(reqs, audit)))
