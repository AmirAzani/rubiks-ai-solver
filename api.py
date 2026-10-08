from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import sys
import os
import json

app = Flask(__name__)
CORS(app)  # Allow your GitHub Pages site to call this API

# The path to the solver script. We'll call it via subprocess for simplicity.
SOLVER_SCRIPT = "solve.py"

@app.route('/health', methods=['GET'])
def health():
    """Simple health check endpoint."""
    return jsonify({"status": "ok"}), 200

@app.route('/solve', methods=['POST'])
def solve_cube():
    """
    Expects JSON: { "scramble": "R U F' ..." }
    or           { "facelets": "UUUUUUUUURRR..." }
    Returns JSON: { "solution": "R U' F2 ..." }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    # Build the command to run solve.py
    cmd = [sys.executable, SOLVER_SCRIPT]

    if "scramble" in data:
        cmd.extend(["--scramble", data["scramble"]])
    elif "facelets" in data:
        cmd.extend(["--facelets", data["facelets"]])
    else:
        return jsonify({"error": "Provide either 'scramble' or 'facelets'"}), 400

    # Optional: allow tuning beam width from the request
    beam = data.get("beam", 64)
    cmd.extend(["--beam", str(beam)])

    try:
        # Run the solver script as a subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )

        if result.returncode != 0:
            return jsonify({
                "error": "Solver failed",
                "stderr": result.stderr
            }), 500

        # The solve.py script prints the solution to stdout.
        # We'll return it as-is.
        solution = result.stdout.strip()

        return jsonify({"solution": solution}), 200

    except subprocess.TimeoutExpired:
        return jsonify({"error": "Solver timed out"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run on port 5000, accessible from other devices on your network
    app.run(host='0.0.0.0', port=5000, debug=True)