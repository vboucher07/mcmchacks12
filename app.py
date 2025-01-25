import frontend

from backend import dummyTest


if __name__ == "__main__":
    app = frontend.create_app()

    @frontend.register_action("previous")
    def previous_action():
        return dummyTest.test_prev()

    @frontend.register_action("next")
    def next_action():
        return dummyTest.test_next()

    app.run(host="0.0.0.0", port=8000, debug=True)
