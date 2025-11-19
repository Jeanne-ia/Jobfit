# Jobfit — prototipo de agente de encaje laboral

Prototipo que, dada una oferta de trabajo (URL o texto) y un CV en texto plano, realiza:

- Extracción heurística de requisitos de la oferta.
- Parseo simple del CV (skills, experiencia, educación).
- Auditoría: para cada requisito, indica si hay evidencia en el CV y devuelve oraciones que prueban el encaje.
- Adaptación del CV: resalta (no inventa) las evidencias encontradas.

Uso rápido:

```bash
python cli.py --job-file oferta.txt --cv-file cv.txt
python cli.py --job-url https://empresa.example/job/123 --cv-file cv.txt --out-adapted cv_adaptado.txt
```

Limitaciones y notas:

- Este es un prototipo con heurísticas claras y transparentes. No usa modelos generativos ni añade información nueva.
- Para producción se recomienda usar análisis semántico con embeddings y OCR/parseo de PDFs/Word.
# Jobfit
Agente inteligente adaptador de cv
