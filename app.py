import frontend

from backend import steps


if __name__ == "__main__":
    app = frontend.create_app()

    @frontend.register_action("prev")
    def previousStep_action():
        image_path, aruco_code = steps.prev()
        print(aruco_code)
        return image_path

    @frontend.register_action("next")
    def nextStep_action():
        image_path, aruco_code = steps.next()
        print(aruco_code)
        return image_path

    @frontend.register_action("update")
    def update_action():
        print("update")
        steps.prev()
        image_path, aruco_code = steps.next()
        return image_path

    app.run(host="0.0.0.0", port=8000, debug=True)
