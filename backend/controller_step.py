from backend import Instructions


currentStep = Instructions.generate_steps()
currentSubstep = currentStep.get_head()


def nextStep():
    global currentStep
    global currentSubstep

    currentStep = currentStep.get_next()
    currentSubstep = currentStep.get_head()

    return currentSubstep.get_path()

def prevStep():
    global currentStep
    global currentSubstep

    currentStep = currentStep.get_prev()
    currentSubstep = currentStep.get_head()

    return currentSubstep.get_path()

def nextSubstep():
    global currentStep
    global currentSubstep

    currentSubstep = currentSubstep.get_next()

    return currentSubstep.get_path()

def prevSubstep():
    global currentStep
    global currentSubstep

    currentSubstep = currentSubstep.get_prev()

    return currentSubstep.get_path()
