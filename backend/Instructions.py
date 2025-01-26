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



#Testing
"""
part1 = Part(1, [1])
part2 = Part(2, [1, 3, 5, 7])
part3 = Part(3, [2,4,6])
part4 = Part(4, [8])
substep1_1_parts = [part1]
substep1_2_parts = [part1, part2]
substep2_1_parts = [part3]
substep1_1 = Substep(1, substep1_1_parts, 0)
substep1_1.add_substep()
substep1_2 = Substep(1, substep1_2_parts, 0)
substep1_2.add_substep()
substep2_1 = Substep(1, substep2_1_parts, 0)
substep2_1.add_substep()
step1 = Step(1, substep1_1)
step2 = Step(2, substep2_1)
step1.add_step()
step2.add_step()

print(step1)
print(step1.substep_head)
print(step1.substep_head.next)
print(step1.next)
print(step2.get_next_step())
"""



