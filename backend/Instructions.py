class Part:

    def __init__(self, tag, step_num, substep_num):
        self.tag = tag
        self.step_num = step_num
        self.substep_num = substep_num

    def __str__(self):
        return f"Part Tag: {self.stag} Steps: {self.step_num} Substeps: {self.substep_num}"

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
        return f"Step: {self.step_num} Parts: {parts_tags}"

    def get_next_substep(self):
        if self.prev == None:
            return self
        return self.next
    
    def get_prev_substep(self):
        if self.next == None:
            return self
        return self.prev

    def add_step(step):
        if num_of_substeps == 0:
            head = step
            tail = step
        else:
            tail.next = step
            tail = step
        num_of_substeps+=1

import Part

class Step: 
    def __init__(self, number, instructions):
        self.number = number
        self.instructions = instructions
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
        return f"Step: {self.step_num}"

    def get_next_step(self):
        if self.prev == None:
            return self
        return self.next
    
    def get_prev_step(self):
        if self.next == None:
            return self
        return self.prev

    def add_step(step):
        if num_of_steps == 0:
            head = step
            tail = step
        else:
            tail.next = step
            tail = step
        num_of_steps+=1