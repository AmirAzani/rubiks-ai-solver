from flask import Flask, request, jsonify
from flask_cors import CORS
import kociemba

app = Flask(__name__)
CORS(app)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200


@app.route('/solve', methods=['POST'])
def solve_cube():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    # --- Case 1: User sends a scramble string ---
    if "scramble" in data:
        scramble = data["scramble"].strip()
        if not scramble:
            return jsonify({"error": "Empty scramble"}), 400
        try:
            # The solution to a scramble (applied to a solved cube)
            # is the inverse of the scramble, reversed.
            solution = inverse_scramble(scramble)
            return jsonify({"solution": solution}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # --- Case 2: User sends a facelets string ---
    facelets = data.get("facelets")
    if not facelets:
        return jsonify({"error": "Provide 'scramble' or 'facelets'"}), 400

    if len(facelets) != 54:
        return jsonify({"error": f"Facelets must be 54 chars, got {len(facelets)}"}), 400

    try:
        solution = kociemba.solve(facelets)
        return jsonify({"solution": solution}), 200
    except Exception as e:
        return jsonify({"error": f"Solver failed: {str(e)}"}), 500


def inverse_scramble(scramble):
    """
    Reverse a scramble string.
    R U F'  →  F U' R'
    Only valid when the scramble was applied to a solved cube.
    """
    moves = scramble.split()
    inverse = []
    for move in reversed(moves):
        if move.endswith("'"):
            inverse.append(move[:-1])       # R' → R
        elif move.endswith("2"):
            inverse.append(move)            # R2 → R2 (self-inverse)
        else:
            inverse.append(move + "'")      # R → R'
    return " ".join(inverse)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)