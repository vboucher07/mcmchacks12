import os


IMAGES_DIRECTORY = "./frontend/static/steps"


class Part:

    def __init__(self, tag, step_num):
        self.tag = tag
        self.step_num = step_num
        #self.substep_num = substep_num

    def __str__(self):
        return f"Part Tag: {self.stag} Steps: {self.step_num}"

class Substep: 
    def __init__(self, number, parts, instructions):
        self.number = number
        self.parts = parts
        self.instructions = instructions
        self.prev = None
        self.next = None
       
    head = None
    tail = None
    global num_of_substeps 
    num_of_substeps = 0
    
    def get_parts_tags(self):
        parts_tags = []
        for i in self.parts:
            parts_tags.append(self.parts[i].tag)
        return parts_tags

    def __str__(self):
        parts_tags = self.get_parts_tags
        return f"Subtep: {self.number} Parts: {parts_tags}"

    def get_next_substep(self):
        if self.next == None:
            return self
        return self.next
    
    def get_prev_substep(self):
        if self.prev == None:
            return self
        return self.prev

    def add_substep(self):
        global num_of_substeps
        if num_of_substeps == 0:
            Substep.head = self
            Substep.tail = self
        else:
            Substep.tail.next = self
            self.prev = Substep.tail
            Substep.tail = self
        num_of_substeps+=1

class Step: 
    def __init__(self, number, substep_head):
        self.number = number
        self.substep_head = substep_head
        self.prev = None
        self.next = None
            
    head = None
    tail = None
    global num_of_steps
    num_of_steps = 0
    
    """def get_parts_tags(self):
        parts_tags = []
        for i in self.parts:
            parts_tags.append(self.parts[i].tag)
        return parts_tags*/"""

    def __str__(self):
        return f"Step: {self.number}"

    def get_next_step(self):
        if self.next == None:
            return self
        return self.next
    
    def get_prev_step(self):
        if self.prev == None:
            return self
        return self.prev

    def add_step(self):
        global num_of_steps
        if num_of_steps == 0:
            Step.head = self
            Step.tail = self
        else:
            Step.tail.next = self
            self.prev = Step.tail
            Step.tail = self
        num_of_steps+=1


def generate_from_images():
    for entry in sorted(os.listdir(IMAGES_DIRECTORY)):
        entry_path = os.path.join(IMAGES_DIRECTORY, entry)
        if os.path.isdir(entry_path):  # Check if it's a directory
            print(entry)


if __name__ == "__main__":
    generate_from_images()
