# Rubik's Cube AI Solver API

A Flask API that solves Rubik's Cubes using a pre-trained PyTorch model.

## Model

Uses [briscoooe/tiny-cube-solver](https://huggingface.co/briscoooe/tiny-cube-solver) — a 33.6M parameter neural network that estimates cube distance, combined with beam search.

## Endpoints

### `GET /health`
Health check.

### `POST /solve`
Body:
```json
{ "scramble": "R U F' D2 L B'" }