from flask import Flask, jsonify

app = Flask(__name__)

@app.get("/")
def home():
    return jsonify({
        "status": "success",
        "message": "SignBridge backend is running!"
    })

@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok"
    })