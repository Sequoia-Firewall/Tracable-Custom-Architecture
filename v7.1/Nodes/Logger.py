import sys

class Logger:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.log_file = 'logs.txt'

    def log(self, message, Loud):
        if self.enabled:
            if Loud == True:
                print(f"[LOG]: {message}", file=sys.stdout)
            with open(self.log_file, 'a') as f:
                f.write(f"{message}\n")