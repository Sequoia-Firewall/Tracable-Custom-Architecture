class Logger:
    def __init__(self, filename, log_level, console_level=1):
        self.logs_folder = "logs/"
        self.filename = filename
        self.classifications = {
            4: "[INFO]: ",             #level 4
            3: "[WARNING]: ",          #level 3
            2: "[ERROR]: ",            #level 2
            1: "[DEBUG]: "             #level 1
        }
        self.log_level = log_level
        self.console_level = console_level  # minimum classification shown on console (1=all, 3=warn+info only)
        self.log_check_count = 10 # every 10 messages the logger will check for repeated
        self._buffer = []  # list of (message, classification, Loud)
        self._call_count = 0
        self._preamble_counts = {}  # preamble -> total count seen
        self._preamble_log_threshold = 1000  # write a condensed log at most every N repeated messages

    def log(self, message, classification, Loud):
        # Check if folder exists, if not create it
        import os
        if not os.path.exists(self.logs_folder):
            os.makedirs(self.logs_folder)

        try:
            file = open(self.logs_folder + self.filename, "a")
        except FileNotFoundError:
            #create the file if it does not exist
            try:
                file = open(self.logs_folder + self.filename, "w")
            except Exception as e:
                print(f"Failed to create log file: {e}")
                return

        except Exception as e:
            print(f"Failed to open log file: {e}")
            return
        if classification not in self.classifications:
            raise ValueError("Invalid classification level.")

        if classification <= 2:  # ERROR and DEBUG — flush immediately
            self.repeated_check(file)  # flush any pending buffered messages first
            prefix = self.classifications[classification]
            formatted_message = f"{prefix}{message}\n"
            if classification <= self.log_level:
                file.write(formatted_message)
            if Loud and classification >= self.console_level:
                print(formatted_message, end='')
        else:
            self._buffer.append((message, classification, Loud))
            self._call_count += 1
            if self._call_count >= self.log_check_count:
                self.repeated_check(file)
                self._call_count = 0

        file.close()

    def repeated_check(self, file):
        if not self._buffer:
            return
        # Group messages by their first 10 characters (preamble)
        from collections import OrderedDict
        groups = OrderedDict()  # preamble -> list of (msg, cls, loud)
        for msg, cls, loud in self._buffer:
            preamble = msg[:10]
            if preamble not in groups:
                groups[preamble] = []
            groups[preamble].append((msg, cls, loud))

        for preamble, entries in groups.items():
            first_msg, first_cls, first_loud = entries[0]
            prefix = self.classifications[first_cls]

            if preamble not in self._preamble_counts:
                # First time seeing this preamble — log it normally
                self._preamble_counts[preamble] = len(entries)
                formatted = f"{prefix}{first_msg}\n"
                if first_cls <= self.log_level:
                    file.write(formatted)
                if first_loud and first_cls >= self.console_level:
                    print(formatted, end='')
            else:
                old_count = self._preamble_counts[preamble]
                new_count = old_count + len(entries)
                self._preamble_counts[preamble] = new_count
                # Only write a condensed log when crossing a 1000-message threshold
                if old_count // self._preamble_log_threshold != new_count // self._preamble_log_threshold:
                    formatted = f"{prefix}{first_msg} (repeated {new_count}x total)\n"
                    if first_cls <= self.log_level:
                        file.write(formatted)
                    if first_loud and first_cls >= self.console_level:
                        print(formatted, end='')
        self._buffer.clear()
