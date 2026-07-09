import os
import dlib
import numpy as np

class LandmarksDetector:
    def __init__(self, predictor_model_path=None):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        
        if predictor_model_path is None:
            cur_dir = os.path.dirname(os.path.realpath(__file__)) 
            predictor_model_path = cur_dir + '/shape_predictor_68_face_landmarks.dat'
        
        self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):

        img = np.array(image)[:,:,0:3]
        
        dets = self.detector(img, 1)

        for detection in dets:
            face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
            yield face_landmarks
