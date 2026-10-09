## Architecture Decision

This project has two implementations:

### Main branch — Kociemba (Production)
Uses the `kociemba` Python library to solve cubes in <1 second.
Runs on Render's free tier (50 MB RAM).
**Live API**: https://rubiks-ai-solver-o9gm.onrender.com

### ml-version branch — PyTorch (Experimental)
Uses a pre-trained neural network (`briscoooe/tiny-cube-solver`) with beam search.
Requires 2 GB+ RAM — does not fit on free-tier hosting.
Kept for reference and research purposes.

**Trade-off**: The ML approach is more impressive academically, but Kociemba
is the correct choice for a public web service where cost, speed, and
reliability matter.