import os
import time
import sys
from gpt4all import GPT4All
from VisionModule import VisionModule
from MotorController import MotorController


class Pikachu:
    COLOR_RESET = "\033[0m"
    COLOR_GREEN = "\033[32m"
    COLOR_RED = "\033[31m"
    COLOR_YELLOW = "\033[33m"

    def __init__(self) -> None:
        self.visionModule = VisionModule()
        self.visionModule.startCapture()
        # self.textModule = TextModule()
        with open(f"Prompts/input to thought.txt", "r") as f:
            self.basicPrompt = f.read()
        self._suppressOutput()
        self.brain = GPT4All(
            "llama-2-7b-chat.ggmlv3.q4_0.bin",
            "/Users/anirudhsingh/MISC/playground/models/",
            allow_download=False,
        )
        self._normalOutput()
        self.motorController = MotorController()

    def _suppressOutput(self):
        self.stdout_fd = os.dup(1)
        self.stderr_fd = os.dup(2)

        # Open a file to /dev/null
        self.devnull_fd = os.open(os.devnull, os.O_WRONLY)

        # Replace stdout and stderr with /dev/null
        os.dup2(self.devnull_fd, 1)
        os.dup2(self.devnull_fd, 2)

        # Writes to sys.stdout and sys.stderr should still work
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = os.fdopen(self.stdout_fd, "w")
        sys.stderr = os.fdopen(self.stderr_fd, "w")

    def _normalOutput(self):
        # Restore stdout and stderr to their original state
        os.dup2(self.stdout_fd, 1)
        os.dup2(self.stderr_fd, 2)

        # Close the saved copies of the original stdout and stderr file descriptors
        os.close(self.stdout_fd)
        os.close(self.stderr_fd)

        # Close the file descriptor for /dev/null
        os.close(self.devnull_fd)

        # Restore sys.stdout and sys.stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    def process(self, cameraInput=None, speechInput=None):
        if cameraInput is None:
            cameraInput = self.visionModule.getAnnotations()
        if speechInput is None:
            speechInput = input(f"{self.COLOR_YELLOW}Speech input: {self.COLOR_RESET}")
        self.setInput(cameraInput, speechInput)
        out = next(self.think())
        return out

    def setInput(self, visionInp=None, speechInp=None):
        input = f"User: [Camera]: {visionInp if visionInp else f'<none>'} [Annotation]: {speechInp if speechInp else f'<none>'}"
        self.input = input

    def setThought(self, thought):
        try:
            if "[start description]" in thought:
                startIdx = thought.index("[start description]") + len(
                    "[start description]"
                )
            if "[start of description]" in thought:
                startIdx = thought.index("[start of description]") + len(
                    "[start of description]"
                )
            endIdx = thought.index("[end description]")
            thought = thought[startIdx:endIdx]
        except Exception as e:
            thought = "stand still"
        thought = f"{thought}"
        self.generatedThought = thought

    def think(self):
        with self.brain.chat_session(system_prompt=self.basicPrompt):
            while True:
                print(f"{self.COLOR_RED}Input to brain: {self.COLOR_RESET}{self.input}")
                output = self.brain.generate(self.input)
                print(f"{self.COLOR_GREEN}Output by brain: {self.COLOR_RESET}{output}")
                yield output

    def awaken(self):
        timePre = time.time()
        self.process("<none>", "<none>")
        timePost = time.time()
        print("###############################################")
        print(f"Time taken to setup: {timePost - timePre:0.2f}")
        print("READY")
        print("###############################################")
        while True:
            # inp = input("Speech Input: ")
            timePre = time.time()
            output = self.process(None, None)
            timePost = time.time()
            print(f"Time Taken: {timePost - timePre : 0.2f}")
            # int(f"{output}")


if __name__ == "__main__":
    timePre = time.time()
    pik = Pikachu()
    pik.awaken()
