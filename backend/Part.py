class Part:

    def __init__(self, tag, step_num):
        self.tag = tag
        self.step_num = step_num

    def __str__(self):
        return f"Step: {self.step_num}"