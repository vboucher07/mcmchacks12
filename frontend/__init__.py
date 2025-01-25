from flask import Flask, render_template

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


def run():
    app.run(host="0.0.0.0", port=8000, debug=True)

if __name__ == "__main__":
    run()
