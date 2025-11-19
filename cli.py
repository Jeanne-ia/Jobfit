#!/usr/bin/env python3
"""CLI para el prototipo Jobfit.

Ejemplos:
  python cli.py --job-file oferta.txt --cv-file cv.txt
  python cli.py --job-url https://empresa.example/careers/123 --cv-file cv.txt

Output: imprime auditoría, justificación y CV adaptado (texto).
"""
import argparse
import sys
from jobfit import analyzer


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Jobfit: audita oferta y adapta CV")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job-url", help="URL de la oferta de trabajo")
    group.add_argument("--job-file", help="Fichero con el texto de la oferta")
    parser.add_argument("--cv-file", required=True, help="Fichero con el CV en texto plano")
    parser.add_argument("--out-adapted", help="Fichero donde guardar CV adaptado")
    parser.add_argument("--use-embeddings", action="store_true", help="Habilitar búsqueda semántica con sentence-transformers + FAISS (local)")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2", help="Nombre del modelo sentence-transformers a usar")
    args = parser.parse_args(argv)

    if args.job_url:
        print(f"Descargando oferta desde: {args.job_url}")
        job_text = analyzer.fetch_job_posting_from_url(args.job_url)
    else:
        job_text = read_text_file(args.job_file)

    cv_text = read_text_file(args.cv_file)

    print("\n--- Extrayendo requisitos de la oferta ---\n")
    reqs = analyzer.extract_requirements(job_text)
    for i, r in enumerate(reqs, 1):
        print(f"{i}. {r}")

    print("\n--- Parseando CV ---\n")
    parsed = analyzer.parse_cv_text(cv_text)
    model = None
    index = None
    sentences = None
    if args.use_embeddings:
        print("Construyendo índice de embeddings (local). Esto usa un modelo pequeño para reducir huella energética.")
        sentences = parsed.get('all_sentences', [])
        model, index, sentences = analyzer.build_embeddings_index(sentences, model_name=args.model_name)

    print(f"Skills detectadas (top 20): {', '.join(parsed.get('skills', [])[:20])}")

    print("\n--- Auditoría (por requisito) ---\n")
    audit = analyzer.audit_requirements(reqs, parsed, use_embeddings=args.use_embeddings, model=model, index=index, sentences=sentences)
    for item in audit:
        print(f"Requisito: {item['requirement']}")
        print(f"  - Score: {item['score']}")
        if item['skills_matches']:
            print(f"  - Skills matches: {', '.join(item['skills_matches'])}")
        if item['evidence_sentences']:
            print(f"  - Evidencia: {item['evidence_sentences'][0]}")
        else:
            print("  - Evidencia: (no encontrada)")
        print()

    print("\n--- Justificación resumida ---\n")
    just = analyzer.justify_fit(reqs, audit)
    for j in just:
        print(j)

    adapted = analyzer.adapt_cv(reqs, parsed, cv_text)
    if args.out_adapted:
        with open(args.out_adapted, "w", encoding="utf-8") as f:
            f.write(adapted)
        print(f"\nCV adaptado guardado en: {args.out_adapted}")
    else:
        print("\n--- CV Adaptado (previsualización) ---\n")
        print(adapted)


if __name__ == "__main__":
    main()
