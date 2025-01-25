import Part

class Step: 
    def __init__(self, number, num_of_parts, parts, instructions):
        self.number = number
        self.num_of_parts = num_of_parts
        self.parts = parts
        self.instructions = instructions
        self.prev = None
        self.next = None
            
    head = None
    tail = None
    global num_of_steps
    num_of_steps = 0
    
    def get_parts_tags(self):
        parts_tags = []
        for i in self.parts:
            parts_tags.append(self.parts[i].tag)
        return parts_tags

    def __str__(self):
        parts_tags = self.get_parts_tags
        return f"Step: {self.step_num} Parts: {parts_tags}"

    def get_next_step(self):
        if self.prev == None:
            return self
        return self.next_step
    
    def get_prev_step(self):
        if self.next == None:
            return self
        return self.prev_step

    def add_step(step):
        if num_of_steps == 0:
            head = step
            tail = step
        else:
            tail.next = step
            tail = step
        num_of_steps+=1
    


    

    

        


