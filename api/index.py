from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/")
def home():
    return jsonify({
        "status": "success",
        "message": "SignBridge backend is running!"
    })


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok"
    })