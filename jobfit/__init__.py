"""Jobfit: herramientas para auditar ofertas y adaptar CVs.

Módulo pequeño con funciones para:
- extraer texto de una oferta (URL o texto plano)
- extraer requisitos de la oferta
- parsear CV en texto (skills, experiencia, educación)
- auditar requisitos contra el CV sin inventar información
- adaptar el CV enfatizando evidencias existentes
- justificar el encaje con citas del CV

Este paquete es un prototipo; las heurísticas son simples y transparentes.
"""

from . import analyzer

__all__ = ["analyzer"]
