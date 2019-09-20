class Controllers(object):
    def __init__(self):
        self.enabled = False

    def disable(self):
        if self.enabled:
            self.enabled = False
            print("Controllers disabled")

    def enable(self):
        if not self.enabled:
            self.enabled = True
            print("Controllers enabled")