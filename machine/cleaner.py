class Cleaner(object):
    def __init__(self):
        self.is_on = False

    def turn_on(self):
        if not self.is_on:
            self.is_on = True
            print("Cleaner turned on")

    def turn_off(self):
        if self.is_on:
            self.is_on = False
            print("Cleaner turned off")
