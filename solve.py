"""
Solve a Rubik's Cube with tiny-cube-solver.

The network does not write moves. It reads one cube and guesses how many moves
it is from solved. This script tries every move from the current positions,
asks the network to score all the results, keeps the best `--beam` of them, and
repeats until one of them is the solved cube. Every move is applied by the code
below, and the answer is checked before it is printed.

    pip install torch transformers safetensors huggingface_hub
    python solve.py --scramble "R U F' D2 L B'"
    python solve.py --facelets UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB

Facelets use the same 54-character layout as the `kociemba` package: faces in
the order U R F D L B, nine stickers each, each sticker named by the face whose
centre has its colour.

Search size (`--beam`) is the one setting that matters:
  64    solves every cube scrambled with 10 moves or fewer; ~25% of full scrambles
  1024  solves ~96% of full scrambles; about 16x the work
A graphics card is strongly recommended above 10 moves.
"""
import argparse
import json
import sys
import time

import torch

# Sticker permutation for one clockwise quarter turn of each face.
QT = {
  "U": [6, 3, 0, 7, 4, 1, 8, 5, 2, 45, 46, 47, 12, 13, 14, 15, 16, 17, 9, 10, 11, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 18, 19, 20, 39, 40, 41, 42, 43, 44, 36, 37, 38, 48, 49, 50, 51, 52, 53],
  "R": [0, 1, 20, 3, 4, 23, 6, 7, 26, 15, 12, 9, 16, 13, 10, 17, 14, 11, 18, 19, 29, 21, 22, 32, 24, 25, 35, 27, 28, 51, 30, 31, 48, 33, 34, 45, 36, 37, 38, 39, 40, 41, 42, 43, 44, 8, 46, 47, 5, 49, 50, 2, 52, 53],
  "F": [0, 1, 2, 3, 4, 5, 44, 41, 38, 6, 10, 11, 7, 13, 14, 8, 16, 17, 24, 21, 18, 25, 22, 19, 26, 23, 20, 15, 12, 9, 30, 31, 32, 33, 34, 35, 36, 37, 27, 39, 40, 28, 42, 43, 29, 45, 46, 47, 48, 49, 50, 51, 52, 53],
  "D": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 24, 25, 26, 18, 19, 20, 21, 22, 23, 42, 43, 44, 33, 30, 27, 34, 31, 28, 35, 32, 29, 36, 37, 38, 39, 40, 41, 51, 52, 53, 45, 46, 47, 48, 49, 50, 15, 16, 17],
  "L": [53, 1, 2, 50, 4, 5, 47, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 19, 20, 3, 22, 23, 6, 25, 26, 18, 28, 29, 21, 31, 32, 24, 34, 35, 42, 39, 36, 43, 40, 37, 44, 41, 38, 45, 46, 33, 48, 49, 30, 51, 52, 27],
  "B": [11, 14, 17, 3, 4, 5, 6, 7, 8, 9, 10, 35, 12, 13, 34, 15, 16, 33, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 36, 39, 42, 2, 37, 38, 1, 40, 41, 0, 43, 44, 51, 48, 45, 52, 49, 46, 53, 50, 47],
}
SOLVED = "".join("URFDLB"[i // 9] for i in range(54))
MOVES = [f + s for f in "URFDLB" for s in ("", "'", "2")]
_TURNS = {"": 1, "'": 3, "2": 2}

# Token ids the network was trained on: [BOS] + 54 sticker tokens + [SEP].
# The sticker ids are not contiguous; they are what the training tokenizer
# produced, and every weight in model.safetensors assumes exactly these.
BOS, SEP, PAD = 1, 2, 0
COLOR_ID = dict(zip("URFDLB", [10, 13, 16, 19, 22, 25]))


def apply_move(state: str, move: str) -> str:
    perm = QT[move[0]]
    for _ in range(_TURNS[move[1:]]):
        state = "".join(state[perm[i]] for i in range(54))
    return state


def apply_sequence(state: str, moves) -> str:
    for m in moves:
        state = apply_move(state, m)
    return state


class ValueNet(torch.nn.Module):
    """A small Llama-shaped encoder over the 54 stickers, averaged, to one number."""

    def __init__(self, hidden, layers, heads, vocab_size=28, max_positions=83):
        super().__init__()
        from transformers import LlamaConfig, LlamaModel
        cfg = LlamaConfig(
            vocab_size=vocab_size, hidden_size=hidden, intermediate_size=hidden * 4,
            num_hidden_layers=layers, num_attention_heads=heads,
            num_key_value_heads=heads, max_position_embeddings=max_positions,
            pad_token_id=PAD,
        )
        self.encoder = LlamaModel(cfg)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, ids):
        h = self.encoder(input_ids=ids).last_hidden_state.mean(dim=1)
        return self.head(h).squeeze(-1)


def load(repo_or_dir="briscoooe/tiny-cube-solver", device="cpu"):
    """Loads the network from a local folder or from the Hugging Face Hub."""
    import os
    from safetensors.torch import load_file
    if os.path.isdir(repo_or_dir):
        cfg_path = os.path.join(repo_or_dir, "config.json")
        weights = os.path.join(repo_or_dir, "model.safetensors")
    else:
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(repo_or_dir, "config.json")
        weights = hf_hub_download(repo_or_dir, "model.safetensors")
    cfg = json.load(open(cfg_path))
    net = ValueNet(cfg["hidden"], cfg["layers"], cfg["heads"],
                   cfg["vocab_size"], cfg["max_positions"])
    net.load_state_dict(load_file(weights))
    return net.to(device).eval()


@torch.no_grad()
def distance(net, states, device, batch_size=1024):
    """The network's guess of moves-to-solved for each state."""
    out = []
    for i in range(0, len(states), batch_size):
        ids = torch.tensor([[BOS] + [COLOR_ID[c] for c in s] + [SEP]
                            for s in states[i:i + batch_size]], device=device)
        out.extend(net(ids).float().clamp(min=0.0).tolist())
    return out


def solve(net, state, device="cpu", beam=64, max_steps=26):
    """Beam search over positions. Returns the solving moves, or None."""
    if state == SOLVED:
        return []
    frontier = [(state, [])]
    seen = {state}
    for _ in range(max_steps):
        children, paths = [], []
        for s, path in frontier:
            for mv in MOVES:
                nxt = apply_move(s, mv)
                if nxt in seen:
                    continue
                if nxt == SOLVED:
                    return path + [mv]
                seen.add(nxt)
                children.append(nxt)
                paths.append(path + [mv])
        if not children:
            return None
        h = distance(net, children, device)
        ranked = sorted(zip(h, children, paths), key=lambda x: x[0])
        frontier = [(c, p) for _, c, p in ranked[:beam]]
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--scramble", help='moves applied to a solved cube, e.g. "R U F\' D2"')
    g.add_argument("--facelets", help="54-character cube state, kociemba layout")
    p.add_argument("--beam", type=int, default=64)
    p.add_argument("--model", default="briscoooe/tiny-cube-solver",
                   help="Hub repo id or local folder")
    args = p.parse_args()

    if args.scramble:
        moves = args.scramble.split()
        bad = [m for m in moves if m not in MOVES]
        if bad:
            sys.exit(f"unknown moves: {bad}; use {' '.join(MOVES)}")
        state = apply_sequence(SOLVED, moves)
    else:
        state = args.facelets.strip()
        if len(state) != 54 or set(state) - set("URFDLB") or any(state.count(c) != 9 for c in "URFDLB"):
            sys.exit("facelets must be 54 characters, nine each of U R F D L B")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load(args.model, device)
    t0 = time.time()
    moves = solve(net, state, device, beam=args.beam)
    secs = time.time() - t0
    if moves is None or apply_sequence(state, moves) != SOLVED:
        print(f"not solved (beam {args.beam}, {secs:.1f}s). Try a larger --beam.")
        sys.exit(1)
    print(" ".join(moves))
    print(f"{len(moves)} moves, {secs:.1f}s on {device}, beam {args.beam}", file=sys.stderr)


if __name__ == "__main__":
    main()
