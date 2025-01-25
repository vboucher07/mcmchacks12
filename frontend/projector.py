from flask import Blueprint, render_template

# Create a blueprint
projector_bp = Blueprint('projector', __name__)

@projector_bp.route("/projector")
def projector():
    return render_template("projector.html")
