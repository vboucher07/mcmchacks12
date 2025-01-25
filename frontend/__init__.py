from flask import Flask, request, jsonify

# Import blueprints
from frontend.home import home_bp
from frontend.projector import projector_bp
from frontend.instructions import instructions_bp


ACTION_HANDLERS = {}

def register_action(action_name):
    def decorator(func):
        ACTION_HANDLERS[action_name] = func
        return func
    return decorator


def create_app():
    app = Flask(__name__)

    @app.route("/backend-action", methods=["POST"])
    def backend_action():
        action = request.json.get("action")
        if action in ACTION_HANDLERS:
            result = ACTION_HANDLERS[action]()  # Call the registered function
            return jsonify({"message": f"{action} executed successfully", "result": result})
        else:
            return jsonify({"error": f"Action '{action}' not registered"}), 400

    # Register blueprints
    app.register_blueprint(home_bp)
    app.register_blueprint(projector_bp)
    app.register_blueprint(instructions_bp)

    return app
