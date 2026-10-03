# AI usage disclosure

Following the transparency practices recommended by JOSS (2026) and pyOpenSci:

* **Original prototype (2025):** generated largely by an AI coding agent on Replit. It contained
  machine-learning "predictions" trained on randomly generated data, randomly generated interface
  residues, fallback mock sequences, and an EGCG table without sources.
* **v0.2 rebuild (October 2026):** carried out with Claude Code (Anthropic), directed and reviewed by
  the maintainer. All simulated components were removed. Every data path calls a public database
  and was checked against live responses, and the scientific behaviour is covered by tests (for
  example, the p53 Phe19/Trp23/Leu26 triad at the MDM2 interface in PDB 1YCR).
* **Scope of AI assistance:** code, tests, documentation and dataset-building scripts.
  Scientific choices (sources, thresholds, caveats) were reviewed by a human before release.

If you find an error, please open an issue. Corrections are very welcome.
