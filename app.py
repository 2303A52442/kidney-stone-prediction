from flask import Flask, render_template, request, jsonify
from predict import predict_image

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Run prediction
    result, score = predict_image(file)

    return jsonify({
        "result": result,
        "score": round(score, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)


