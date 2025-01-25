from flask import Flask, render_template, jsonify

app = Flask(__name__)

# Home page
@app.route("/")
def home():
    return render_template("home.html")

# Projector page
@app.route("/projector")
def projector():
    return render_template("projector.html")

# Instructions page
@app.route("/instructions")
def instructions():
    return render_template("instructions.html")

# Previous button action
@app.route("/previous", methods=["POST"])
def previous():
    # Python function logic for "Previous"
    print("Previous button clicked!")
    return jsonify({"message": "Previous function executed successfully"})

# Next button action
@app.route("/next", methods=["POST"])
def next():
    # Python function logic for "Next"
    print("Next button clicked!")
    return jsonify({"message": "Next function executed successfully"})

def run():
    app.run(host="0.0.0.0", port=8000, debug=True)

if __name__ == "__main__":
    run()
