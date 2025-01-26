import os


IMAGES_DIRECTORY = "./frontend/static/steps"


class Substep: 
    def __init__(self, path):
        self.aruco = None
        self.path = path
        self.prev = None
        self.next = None
       
    
    def get_next(self):
        if self.next == None:
            return self
        return self.next
    
    def get_prev(self):
        if self.prev == None:
            return self
        return self.prev

    def set_aruco(self, aruco):
        self.aruco = aruco
        return

    def get_aruco(self):
        return self.aruco

    def get_path(self):
        return self.path

    def set_next(self, nextNode):
        self.next = nextNode
        self.next.prev = self
        return


class Step: 
    def __init__(self, number: int, path):
        self.number = number    #number corresponds to the step number
        self.path = path
        self.substepHead = None
        self.next = None
        self.prev = None
        return

    def get_number(self):
        return self.number

    def get_path(self):
        return self.path
            
    def get_next(self):
        if self.next == None:
            return self
        return self.next
    
    def get_prev(self):
        if self.prev == None:
            return self
        return self.prev

    def set_next(self, nextNode):
        self.next = nextNode
        self.next.prev = self
        return

    def set_head(self, substepHead):
        self.substepHead = substepHead
        return

    def get_head(self):
        return self.substepHead


def generate_substeps(step):
    headSubstep = Substep("")
    previousSubstep = headSubstep
    step_path = step.get_path()

    images = sorted(os.listdir(step_path))
    for image in images:
        image_path = os.path.join(step_path, image)
        currentSubstep = Substep(image_path.replace("./frontend", ""))
        if headSubstep.get_path() == "":
            headSubstep = currentSubstep
            previousSubstep = currentSubstep

        else:
            previousSubstep.set_next(currentSubstep)
            previousSubstep = currentSubstep

    step.set_head(headSubstep)

    return


def generate_steps():
    headStep = Step(-1, "")
    previousStep = headStep
    directories = []

    # Generate list of directories in static/steps
    for entry in sorted(os.listdir(IMAGES_DIRECTORY)):
        entry_path = os.path.join(IMAGES_DIRECTORY, entry)
        if os.path.isdir(entry_path):  # Check if it's a directory
            directories.append(entry_path)

    # Transform array list into linked list, with numbers and paths
    for i in range(len(directories)):
        currentStep = Step(i, directories[i])
        generate_substeps(currentStep)

        if headStep.get_number() == -1:
            headStep = currentStep
            previousStep = currentStep

        else:
            previousStep.set_next(currentStep)
            previousStep = currentStep

    return headStep


if __name__ == "__main__":
    head = generate_steps()
    while(head.next != None):
        print(head.get_number())
        subHead = head.get_head()
        print(subHead.get_path())
        while(True):
            subHead = subHead.get_next()
            print(subHead.get_path())
            if subHead.next == None:
                break
        head = head.get_next()

    while(head.prev != None):
        print(head.get_number())
        head = head.get_prev()
