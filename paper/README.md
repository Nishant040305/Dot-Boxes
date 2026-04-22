# Paper (LaTeX)

This folder contains a self-contained LaTeX project for writing a research paper
about the C++ AlphaZero-style agent in `src/cpp/`.

## Build

From the repo root:

```bash
latexmk -pdf -cd paper/main.tex
```

Clean:

```bash
latexmk -C -cd paper/main.tex
```

