
import numpy as np

class BVTFaceDetector:

    def __init__(self):
        import BVT
        self.det = BVT.Engine()
        self.det.init_humanface_module(faceDetection=True, faceLandmark=True, advancedLandmark=True, attribute=True, iris=True)

    def get_landmarks(self, img, landmark=True, advancedLandmark=False, iris=False):
        faces = self.det.get_face(np.array(img), det_interval=1, run_mode=0, run_level="platinum")

        if landmark:
            landmarks_all_faces = [face.landmark for face in faces]

        if advancedLandmark:
            advanced_landmarks_all_faces = [face.advancedLandmark for face in faces]

        if iris:
            iris_left_landmarks_all_faces = [face.leftIrisLandmark for face in faces]
            iris_right_landmarks_all_faces = [face.rightIrisLandmark for face in faces]

        result = []

        if landmark:
            result.append(landmarks_all_faces)
        if advancedLandmark:
            result.append(advanced_landmarks_all_faces)
        if iris:
            result.append(iris_left_landmarks_all_faces)
            result.append(iris_right_landmarks_all_faces)

        if len(result) == 0:
            return None
        elif len(result) == 1:
            return result[0]
        else:
            return result


    def get_gender(self, img):
        faces = self.det.get_face(np.array(img), det_interval=1, run_mode=0, run_level="platinum")

        gender_all_faces = ['man' if face.attribute[0] > 0.5 else 'woman' for face in faces]

        if len(gender_all_faces) == 0:
            return ['unknown']

        return gender_all_faces

    def get_head_pose(self, img):
        faces = self.det.get_face(np.array(img), det_interval=1, run_mode=0, run_level="platinum")
        head_pose_all_faces = [[face.pitch, face.yaw, face.roll] for face in faces]
        return head_pose_all_faces

