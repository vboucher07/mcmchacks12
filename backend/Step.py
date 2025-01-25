class Step: 
    def __init__(self, number, prev, next, num_of_parts, parts):
        self.number = number
        self.num_of_parts = num_of_parts
        self.parts = parts
        self.prev = None
        self.next = None
                
    head = None
    tail = None
    global num_of_steps
    num_of_steps = 0

    def get_next_step(self):
        return self.next_step
    
    def get_prev_step(self):
        return self.prev_step

    def add_step(step):
        if num_of_steps == 0:
            head = step
            tail = step
        else:
            temp = tail
            tail.next = step
            tail = step
        num_of_steps+=1
    

    

        


