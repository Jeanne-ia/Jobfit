"""App Gradio para revisar auditoría extractiva y confirmar evidencias.

Flujo:
- Usuario pega texto de oferta o URL.
- Usuario pega CV en texto (o sube archivo .txt).
- Ejecutar análisis -> muestra requisitos, justificaciones y lista numerada de evidencias.
- Usuario selecciona índices de evidencias a incluir (por defecto se preseleccionan las top N).
- Generar CV final que resalta solo las evidencias confirmadas.

Nota: la app es extractiva; no reescribe ni inventa información.
"""
import gradio as gr
from jobfit import analyzer
import os
import tempfile
from fpdf import FPDF


def _read_uploaded_file(file) -> str:
    if not file:
        return ""
    # file is a tempfile-like object from gradio
    try:
        # support PDF
        name = file.name
        if name.lower().endswith('.pdf'):
            txt = analyzer.extract_text_from_pdf(name)
            if txt:
                return txt
            # fallback to reading bytes as text
        with open(name, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return ""


def analyze(job_text, job_url, cv_text, cv_file, use_embeddings, model_name):
    # obtener job_text desde url si procede
    if job_url and not job_text:
        try:
            job_text = analyzer.fetch_job_posting_from_url(job_url)
        except Exception as e:
            return f"Error descargando URL: {e}", "", "", ""

    if cv_file and not cv_text:
        cv_text = _read_uploaded_file(cv_file)

    reqs = analyzer.extract_requirements(job_text)
    parsed = analyzer.parse_cv_text(cv_text)
    model = None
    index = None
    sentences = None
    if use_embeddings:
        sentences = parsed.get('all_sentences', [])
        model, index, sentences = analyzer.build_embeddings_index(sentences, model_name=model_name)

    audit = analyzer.audit_requirements(reqs, parsed, use_embeddings=use_embeddings, model=model, index=index, sentences=sentences)

    # realism score
    realism = analyzer.estimate_realism_of_job_posting(job_text)

    # construir lista de evidencias numerada (deduplicada)
    evidences = []
    for item in audit:
        for e in item.get('evidence_sentences', []):
            if e not in evidences:
                evidences.append(e)

    evidence_text = "\n\n".join([f"[{i}] {e}" for i, e in enumerate(evidences, 1)])

    # justificaciones resumidas
    justs = analyzer.justify_fit(reqs, audit)
    just_text = "\n\n".join(justs)

    # preview adaptado: resaltar todas las evidencias encontradas
    adapted_preview = analyzer.adapt_cv(reqs, parsed, cv_text)

    # default selection: top N evidences (all)
    default_selection = ",".join(str(i) for i in range(1, min(6, len(evidences)) + 1)) if evidences else ""

    # also return realism score
    return "\n\n".join(reqs), evidence_text, just_text, adapted_preview, default_selection, realism


def generate_final_cv(selected_indices, original_cv_text, evidence_text):
    # parsear selected_indices como lista de enteros
    if not evidence_text:
        return "(no hay evidencias)"
    evidences = [line.split('] ', 1)[1] for line in evidence_text.split('\n\n') if '] ' in line]
    picks = []
    try:
        for part in selected_indices.split(','):
            p = part.strip()
            if not p:
                continue
            i = int(p)
            if 1 <= i <= len(evidences):
                picks.append(evidences[i-1])
    except Exception:
        return "Índices inválidos. Introduce una lista separada por comas, p.ej. 1,2,4"

    adapted = analyzer.generate_adapted_cv_from_selected_evidence(original_cv_text, picks)
    # limpiar marcas [MATCH]
    clean = adapted.replace('[MATCH]', '').strip()

    # preparar nombre de fichero usando el primer non-empty line como nombre si posible
    first_line = "CV_Adaptado"
    for ln in clean.splitlines():
        t = ln.strip()
        if t:
            # usar la primera línea como nombre (limitado y sanitized)
            first_line = re.sub(r"[^A-Za-z0-9_\- ]", "", t).strip().replace(' ', '_')[:50]
            break

    filename_prefix = f"CV_Adaptado_{first_line}"

    # escribir a fichero PDF temporal para descarga con estilo ATS-friendly
    try:
        fd, path = tempfile.mkstemp(prefix=filename_prefix + '_', suffix='.pdf')
        os.close(fd)
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Estilos solicitados:
        # - Fuente general: Times
        # - Nombre/Títulos: 14pt, Negrita, color celeste (light blue)
        # - Cuerpo: 11pt, color negro
        # Color celeste (light blue)
        light_blue = (173, 216, 230)

        # Nombre / encabezado: tomar la primera línea no vacía del CV
        lines = [l.rstrip() for l in clean.splitlines()]
        name = None
        rest_lines = []
        for l in lines:
            if name is None and l.strip():
                name = l.strip()
            else:
                rest_lines.append(l)
        if name is None:
            name = "CV Adaptado"

        # Nombre grande
        try:
            pdf.set_font('Times', 'B', 14)
        except Exception:
            pdf.set_font('times', 'B', 14)
        pdf.set_text_color(*light_blue)
        pdf.cell(0, 10, name, ln=1)

        # separación
        pdf.ln(2)

        # Cuerpo: iterar y aplicar reglas de formateo
        pdf.set_text_color(0, 0, 0)
        body_font = ('Times', '', 11)
        try:
            pdf.set_font(*body_font)
        except Exception:
            pdf.set_font('times', '', 11)

        for raw in rest_lines:
            line = raw.strip()
            if not line:
                pdf.ln(3)
                continue

            # Section title heuristic: short line ending with ':' or a small line in all-caps or starts with a capitalized word and few words
            is_title = False
            if line.endswith(':') or (len(line.split()) <= 5 and line.isupper()):
                is_title = True
            # also if line looks like 'Profile' or 'Skills' etc
            if not is_title and re.match(r"^(Profile|Skills|Experience|Education|Education:|Experience:|Profile:|Summary:|Skills:)", line, re.I):
                is_title = True

            if is_title:
                # title: bold, light blue, 14pt
                try:
                    pdf.set_font('Times', 'B', 14)
                except Exception:
                    pdf.set_font('times', 'B', 14)
                pdf.set_text_color(*light_blue)
                title_text = line.rstrip(':')
                pdf.multi_cell(0, 8, title_text)
                pdf.ln(1)
                # volver a cuerpo
                pdf.set_text_color(0, 0, 0)
                try:
                    pdf.set_font('Times', '', 11)
                except Exception:
                    pdf.set_font('times', '', 11)
                continue

            # bullets: if line starts with '-' or '*' or '•' or begins with a dash number
            if re.match(r"^[-*•]\s+", line) or re.match(r"^[0-9]+[\).]\s+", line):
                # render bullet with indent
                bullet = '• '
                content = re.sub(r"^([-*•\s]*|[0-9]+[\).]\s*)", '', line)
                pdf.cell(6)
                pdf.multi_cell(0, 7, bullet + content)
            else:
                # regular paragraph, ensure wrap and spacing
                pdf.multi_cell(0, 7, line)

        # guardar PDF
        pdf.output(path)
        return adapted, path
    except Exception:
        return adapted, None
