from flask import Blueprint, render_template, jsonify, request

# Create a blueprint
instructions_bp = Blueprint('instructions', __name__)

@instructions_bp.route("/instructions")
def instructions():
    return render_template("instructions.html")

@instructions_bp.route("/previous", methods=["POST"])
def previous():
    print("Previous button clicked!")
    return jsonify({"message": "Previous function executed successfully"})

@instructions_bp.route("/next", methods=["POST"])
def next():
    print("Next button clicked!")
    return jsonify({"message": "Next function executed successfully"})
