import os
from flask import Flask, render_template, request, jsonify
from smart_agent import SmartCampusAssistant

app = Flask(__name__)

# Set your API key here or via terminal environment variable: export API_KEY="your_key"
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")

assistant = SmartCampusAssistant(
    data_path="data/campus_faqs.json",
    api_key=API_KEY
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query", "").strip()
    engine = data.get("engine", "Sentence-Transformers")

    if not query:
        return jsonify({"error": "Empty query"}), 400

    result = assistant.ask(user_prompt=query, active_engine=engine)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
