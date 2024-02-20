import multiprocessing
import cv2
from cv2 import VideoCapture
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from collections import deque
import time

class SpeechModule():
    def __init__(self, cam_port=0, captureInterval=2):
        self.cam_port = cam_port
        self.maxBufferSize = 10
        self.annBuffer = multiprocessing.Queue(maxsize=self.maxBufferSize)
        self.annBufferSize = 0
        self.stop_event = multiprocessing.Event()  # Event to signal child process to stop
        self.captureInterval = captureInterval

    def captureAndAnnotate(self):
        self.cam = VideoCapture(self.cam_port) 
        while not self.stop_event.is_set():
            _, image = self.cam.read()
            # print(type(image))
            # cv2.imwrite("image.png", image)
            inputs = self.processor(image, self.textPrompt, return_tensors="pt")
            out = self.model.generate(**inputs)
            # if self.annBuffer.qsize() >= self.maxBufferSize: 
            if self.annBufferSize >= self.maxBufferSize:
                self.annBufferSize -= 1
                self.annBuffer.get()
            self.annBufferSize += 1
            annotation = self.processor.decode(out[0], skip_special_tokens=True).replace("a photography of a ", "")
            self.annBuffer.put(annotation)
            time.sleep(self.captureInterval)
        
        self.cam.release()
    
    def startCapture(self):
        self.capturing = True
        self.process = multiprocessing.Process(target=self.captureAndAnnotate)
        self.process.start()

    def stopCapture(self):
        if self.process is not None:
            print("stopping capture")
            self.stop_event.set()
            self.process.join()
            self.process = None

    def getAnnotations(self):
        # print(self.annBuffer)
        annotations = []
        while not self.annBuffer.empty():
            self.annBufferSize -= 1
            annotations.append(self.annBuffer.get())
        annotations = ". ".join(annotations)
        return annotations


if __name__ == "__main__":
    preTime = time.time() 
    visionModule = VisionModule(0)
    ###########################
    ## Single image test
    ###########################
    # postTime = time.time()
    # print(f"Time to initialize: {postTime - preTime:0.2f}")
    # preTime = time.time()
    # visionModule.captureAndAnnotate()
    # postTime = time.time()
    # print(f"time taken: {postTime - preTime:0.2f}")
    # print(f"{visionModule.getAnnotations()}")

    ###########################
    ## Continuous mode test
    ###########################
    visionModule.startCapture()
    time.sleep(10)
    print(f"annotations after 10 sec: {visionModule.getAnnotations()}")
    print(f"annotations after 0 sec: {visionModule.getAnnotations()}")
    time.sleep(5)
    print(f"annotations after 5 sec: {visionModule.getAnnotations()}")
    visionModule.stopCapture()
