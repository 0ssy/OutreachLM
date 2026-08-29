# Phase G — CPU-Native Language Architecture Research

Rules:

1. CPU only.
2. No GPU.
3. No pretrained weights.
4. No external foundational models.
5. No llama.cpp.
6. No GGUF.
7. No Transformers.
8. No pretrained embeddings.
9. NumPy is permitted.
10. Python standard library is permitted.
11. Every parameter must be created from scratch.
12. Every experiment must be reproducible.
13. Every model must have a measurable baseline.
14. Architecture claims must be supported by ablations.
15. Resource usage must be measured.

Clarification:

PyTorch should not be used for the G-series model mathematics. NumPy is the numerical engine so we can see exactly what the architecture is doing.
