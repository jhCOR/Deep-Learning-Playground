import cv2
import numpy as np

class VideoManager():
    def __init__(self, size=(40, 30)):
        self.size = size
        self.step = "init"
        self.data = None

    def reshape(self, content, size):
        content = np.reshape(content, size)
        content = cv2.normalize(content, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        self.data = content
        self.step = "reshape"
        return self

    def makeVideoData(self):
        assert self.data.ndim > 2, "matrix dimension could be larger than 2"
        assert self.data is not None, "excute reshape method first"

        frame_list = []
        for frame in self.data :
            frame_list.append(frame)

        self.data = frame_list
        self.step = "makeVideoData"
        return self
    
    def saveVideo(self, filename="test.avi", frames=10, size=(40, 30)):

        if (self.step == "makeVideoData") | (self.step == "reshape"):
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            videoobject = cv2.VideoWriter(filename, fourcc, frames, (self.size[0], self.size[1]), False)
            
            
            for frame in self.data:
                image = cv2.GaussianBlur(frame, (10, 10), 0)
                _, thresholded_image = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY_INV)
                image_no_background = cv2.bitwise_and(frame, frame, mask=thresholded_image)
                videoobject.write(image_no_background)

            videoobject.release()