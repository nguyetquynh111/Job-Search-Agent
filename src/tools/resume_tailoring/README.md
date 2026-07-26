# Resume Tailoring Tool
This tool creates a job-specific LaTeX resume from the original resume, fit
analysis, and approved evidence.

- See [INPUT.md](INPUT.md) for input fields.
- See [OUTPUT.md](OUTPUT.md) for output fields.
- Contract: `TailorResumeInput` to `TailorResumeOutput`.

Implementation notes:

- Keeps the original resume unchanged and writes each tailored artifact to
  `outputs/<job_id>/`.
- Edits only the template's explicit `AGENT-EDIT-TARGET` and
  `AGENT-SWAP-TARGET` locations.
- Rewrites the professional summary, modifies exactly two experience bullets,
  surfaces only evidenced skills, and applies a project swap only when the fit
  analysis recommends a portfolio-backed project.
- Compiles with `tectonic` when available (self-contained, cross-platform, no
  system LaTeX install required), falling back to `pdflatex` otherwise, and
  verifies the generated PDF is exactly one page. If the first compile
  overflows, it tightens spacing/font size and recompiles.
- Returns a before/after change log with evidence IDs for every edit.
