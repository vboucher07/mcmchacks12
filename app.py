import frontend

from backend import dummyTest
from backend import controller_step


if __name__ == "__main__":
    app = frontend.create_app()

    @frontend.register_action("prevStep")
    def previousStep_action():
        return controller_step.prevStep()

    @frontend.register_action("nextStep")
    def nextStep_action():
        return controller_step.nextStep()

    @frontend.register_action("nextSubstep")
    def nextSubstep_action():
        return controller_step.nextSubstep()

    @frontend.register_action("prevSubstep")
    def prevSubstep_action():
        return controller_step.prevSubstep()

    app.run(host="0.0.0.0", port=8000, debug=True)
