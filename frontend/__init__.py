from flask import Flask

from home import home_bp
from projector import projector_bp
from instructions import instructions_bp


app = Flask(__name__)


def run():
    app.register_blueprint(home_bp)
    app.register_blueprint(projector_bp)
    app.register_blueprint(instructions_bp)

    app.run(host="0.0.0.0", port=8000, debug=True)

if __name__ == "__main__":
    run()
