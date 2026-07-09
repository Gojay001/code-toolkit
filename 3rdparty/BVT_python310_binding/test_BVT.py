# encoding=utf8
"""
    Author   : Bigos Vision Team
    Date     : 2020-03-21
    Describe : Example for Python Binding of Bigo's Vision Toolchain
               more detailed interface documentation in http://wiki.bigo.sg:8090/pages/viewpage.action?pageId=179994682
"""
import BVT
import cv2
import os
import numpy as np
import logging
import unittest
import site
import time

head = '[%(asctime)-15s %(levelname)s] %(filename)s (Line %(lineno)d)] %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)
PROJ_ROOT = os.path.dirname(__file__)
TEST_DATA = os.path.join(PROJ_ROOT, '.')
# you can set model_dir to None or "yourself_model_dir"
# model_dir = os.path.join(os.path.dirname(os.__file__), 'site-packages/BVTData/bvtMobile/models')
model_dir = None
is_viz = False

# DynamicExpression
EYE_BLINK = 0
MOUTH_AH = 1
HEAD_YAW = 2
HEAD_PITCH = 3
BROW_JUMP = 4


def draw_dynamic_expression(img, text_position, expression):
    count = 0
    gap = 25
    gap_to_point0 = 120
    font_size = 0.5
    if expression[EYE_BLINK]:
        cv2.putText(img, "EYE_BLINK", (text_position[0] - gap_to_point0, text_position[1] + count * gap),
                    cv2.FONT_HERSHEY_COMPLEX, font_size, (0, 0, 255))
        count += 1
    else:

        cv2.putText(img, "EYE_BLINK", (text_position[0] - gap_to_point0, text_position[1] + count * gap),
                    cv2.FONT_HERSHEY_COMPLEX, font_size, (255, 255, 255), 1)
        count += 1

    if expression[MOUTH_AH]:
        cv2.putText(img, "MOUTH_AH", (text_position[0] - gap_to_point0, text_position[1] + count * gap),
                    cv2.FONT_HERSHEY_COMPLEX, font_size, (0, 0, 255))
        count += 1
    else:
        cv2.putText(img, "MOUTH_AH", (text_position[0] - gap_to_point0, text_position[1] + count * gap),
                    cv2.FONT_HERSHEY_COMPLEX, font_size, (255, 255, 255))
        count += 1

    if expression[HEAD_YAW]:
        cv2.putText(img, "HEAD_YAW", (text_position[0] - gap_to_point0, text_position[1] + count * gap),
                    cv2.FONT_HERSHEY_COMPLEX, font_size, (0, 0, 255))
        count += 1
    else:
        cv2.putText(img, "HEAD_YAW", (text_position[0] - gap_to_point0, text_position[1] + count * gap),
                    cv2.FONT_HERSHEY_COMPLEX, font_size, (255, 255, 255))
        count += 1

    if expression[HEAD_PITCH]:
        cv2.putText(img, "HEAD_PITCH", (text_position[0] - gap_to_point0, text_position[1] + count * gap),
                    cv2.FONT_HERSHEY_COMPLEX, font_size, (0, 0, 255))
        count += 1
    else:
        cv2.putText(img, "HEAD_PITCH", (text_position[0] - gap_to_point0, text_position[1] + count * gap),
                    cv2.FONT_HERSHEY_COMPLEX, font_size, (255, 255, 255))
        count += 1

    if expression[BROW_JUMP]:
        cv2.putText(img, "BROW_JUMP", (text_position[0] - gap_to_point0, text_position[1] + count * gap),
                    cv2.FONT_HERSHEY_COMPLEX, font_size, (0, 0, 255))
        count += 1
    else:
        cv2.putText(img, "BROW_JUMP", (text_position[0] - gap_to_point0, text_position[1] + count * gap),
                    cv2.FONT_HERSHEY_COMPLEX, font_size, (255, 255, 255))
        count += 1


class BvtInfoTest(unittest.TestCase):

    def test_print_bvt_info(self):
        logging.info('bvt version: %s', BVT.__version__)
        self.assertTrue(True)


class FaceModuleTest(unittest.TestCase):

    def test_face_module_general(self):
        logging.debug('Face module general test')
        # NOTE : if you can't run site.getsitepackages(), you need add argument model_dir="bvt model dir"
        engine = BVT.Engine(model_dir=model_dir)
        engine.setBvtVersion(2)


        # NOTE : by default only open faceDetection, faceLandmark, headPose ``` engine.init_humanface_module() ```
        status = engine.init_humanface_module(faceDetection=True,
                                              faceLandmark=True,
                                              advancedLandmark=True,
                                              iris=True,
                                              tongue=True,
                                              forehead=True,
                                              attribute=True,
                                              expression=True,
                                              headPose=True)
        if not status:
            self.assertTrue(status)
            logging.error('Init face module failed')
            return
        image = cv2.imread("human_face_group_image_3x768x1024.jpg")
        demo_image = image.copy()

        # NOTE : BVT python biding only support PIXEL_RGB so you need to convert color by yourself
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logging.debug("image shape : %s", image.shape)

        # NOTE : 0 for image mode, 1 for video mode;  Not recommended for mixed use image_mode and video_mode
        # run_level : diamond, platinum, gold, silver, bronze, iron
        # output result is a list of detected faces, format is: [{face_0}, {face_1}, {face_2}, ..., {face_n}]
        res = engine.get_face(image, det_interval=1, run_mode=0, run_level="platinum")
        self.assertEqual(len(res), 10)

        for face_index, face in enumerate(res):
            # NOTE : face == res[face_index]
            # for key in face.keys():
            #     logging.debug('%s : %s', key, face[key])
            face_id = face.id
            prob = face.prob
            bbox = [face.x, face.y, face.width, face.height]
            landmark = face.landmark
            landmark_visibility = face.landmarkVisibility
            head_pose = (face.pitch, face.yaw, face.roll)
            tongue_score = face.tongueScore
            # 240 landmark
            advanced_landmark = face.advancedLandmark
            # 20 left iris landmark
            left_iris_landmark = face.leftIrisLandmark
            # 20 right iris landmark
            right_iris_landmark = face.rightIrisLandmark
            left_iris_visibility = face.leftIrisVisibility
            right_iris_visibility = face.rightIrisVisibility
            # 33 forehead landmark
            forehead = face.forehead
            # 7 attribute prob : Male, Yellow, Black, White, Indian, Age, Beauty
            attribute = face.attribute
            # 17 static expression
            static_expression = face.staticExpression
            # 5 dynamic expression
            dynamic_expression = face.dynamicExpression
            logging.debug('id: %d, bbox: %s , head pose : %s', face_id, bbox, head_pose)
            for pt in landmark:
                x = int(pt[0])
                y = int(pt[1])
                cv2.circle(demo_image, center=(x, y), radius=3, color=(255, 0, 0), thickness=-1)
        if is_viz:
            cv2.imshow("demo", demo_image)
            cv2.waitKey()

    def test_face_module_general_video_mode(self):
        logging.debug('Face module video mode general test')
        # NOTE : if you can't run site.getsitepackages(), you need add argument model_dir="bvt model dir"
        engine = BVT.Engine(model_dir=model_dir)
        engine.setBvtVersion(2)

        # NOTE : by default only open faceDetection, faceLandmark, headPose ``` engine.init_humanface_module() ```
        status = engine.init_humanface_module(faceDetection=True,
                                              faceLandmark=True,
                                              advancedLandmark=True,
                                              iris=True,
                                              tongue=True,
                                              forehead=True,
                                              attribute=True,
                                              expression=True,
                                              headPose=True)
        if not status:
            self.assertTrue(status)
            logging.error('Init face module failed')
            return
        video_path = '3148a34d-a268-4dc8-9132-d87f0b0b3133.mp4'
        if not os.path.exists(video_path):
            logging.error('video is not existed! %s', video_path)
        cap = cv2.VideoCapture(video_path)
        while True:
            res, image = cap.read()
            if not res:
                break
            demo_image = image.copy()

            # NOTE : BVT python biding only support PIXEL_RGB so you need to convert color by yourself
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logging.debug("image shape : %s", image.shape)

            # NOTE : 0 for image mode, 1 for video mode  Not recommended for mixed use image_mode and video_mode
            # run_level : diamond, platinum, gold, silver, bronze, iron
            # output result is a list of detected faces, format is: [{face_0}, {face_1}, {face_2}, ..., {face_n}]
            res = engine.get_face(image, det_interval=-1, run_mode=1, run_level="platinum")

            for face_index, face in enumerate(res):
                # NOTE : face == res[face_index]
                # for key in face.keys():
                #     logging.debug('%s : %s', key, face[key])
                face_id = face.id
                prob = face.prob
                bbox = [face.x, face.y, face.width, face.height]
                landmark = face.landmark
                landmark_visibility = face.landmarkVisibility
                head_pose = (face.pitch, face.yaw, face.roll)
                tongue_score = face.tongueScore
                # 240 landmark
                advanced_landmark = face.advancedLandmark
                # 20 left iris landmark
                left_iris_landmark = face.leftIrisLandmark
                # 20 right iris landmark
                right_iris_landmark = face.rightIrisLandmark
                left_iris_visibility = face.leftIrisVisibility
                right_iris_visibility = face.rightIrisVisibility
                # 33 forehead landmark
                forehead = face.forehead
                # 7 attribute prob : Male, Yellow, Black, White, Indian, Age, Beauty
                attribute = face.attribute
                # 17 static expression
                static_expression = face.staticExpression
                # 5 dynamic expression
                dynamic_expression = face.dynamicExpression
                logging.debug('id: %d, bbox: %s , head pose : %s', face_id, bbox, head_pose)
                if is_viz:
                    draw_dynamic_expression(demo_image, (demo_image.shape[1], 20), dynamic_expression)
                for pt in landmark:
                    x = int(pt[0])
                    y = int(pt[1])
                    cv2.circle(demo_image, center=(x, y), radius=3, color=(255, 0, 0), thickness=-1)
            if is_viz:
                cv2.imshow("demo", demo_image)
                cv2.waitKey(30)

    def test_face_module_liveness_detection(self):
        logging.debug('Face module liveness detection test')
        # NOTE : if you can't run site.getsitepackages(), you need add argument model_dir="bvt model dir"
        engine = BVT.Engine(model_dir=model_dir)

        # NOTE : by default only open faceDetection, faceLandmark, headPose ``` engine.init_humanface_module() ```
        status = engine.init_humanface_module(faceDetection=True,
                                              faceLandmark=True,
                                              advancedLandmark=True,
                                              headPose=True,
                                              livenessDetection=True)
        if not status:
            self.assertTrue(status)
            logging.error('Init face module failed')
            return
        image = cv2.imread("liveness_detection_normalImg_3x814x914.jpg")
        demo_image = image.copy()

        ir_image = cv2.imread("liveness_detection_thermalImg_3xx814x914.jpg")
        ir_demo_image = image.copy()

        # NOTE : BVT python biding only support PIXEL_RGB so you need to convert color by yourself
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ir_image = cv2.cvtColor(ir_image, cv2.COLOR_BGR2RGB)
        logging.debug("image shape : %s", image.shape)

        # NOTE : 0 for image mode, 1 for video mode;  Not recommended for mixed use image_mode and video_mode
        # run_level : diamond, platinum, gold, silver, bronze, iron
        # output result is a list of detected faces, format is: [{face_0}, {face_1}, {face_2}, ..., {face_n}]
        res = engine.get_face(image, det_interval=1, run_mode=0, run_level="platinum", extra_image_data=ir_image)
        print(len(res))
        self.assertEqual(len(res), 2)

        for face_index, face in enumerate(res):
            # NOTE : face == res[face_index]
            # for key in face.keys():
            #     logging.debug('%s : %s', key, face[key])
            face_id = face.id
            prob = face.prob
            bbox = [face.x, face.y, face.width, face.height]
            landmark = face.landmark
            landmark_visibility = face.landmarkVisibility
            head_pose = (face.pitch, face.yaw, face.roll)
            # 1 liveness score
            liveness_score = face.liveness
            logging.debug('id: %d, bbox: %s , head pose : %s, liveness: %f', face_id, bbox, head_pose, liveness_score)

            cv2.putText(demo_image, "id: %d" % face_id, (int(landmark[0][0]), int(landmark[0][1] - 20)),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            for pt in landmark:
                x = int(pt[0])
                y = int(pt[1])
                cv2.circle(demo_image, center=(x, y), radius=3, color=(255, 0, 0), thickness=-1)
        if is_viz:
            cv2.imshow("demo", demo_image)
            cv2.waitKey()

    def test_face_module_with_external_detection(self):
        logging.debug('Face module test with external detection')
        # NOTE : if you can't run site.getsitepackages(), you need add argument model_dir="bvt model dir"
        engine = BVT.Engine(model_dir=model_dir)

        # NOTE : by default only open faceDetection, faceLandmark, headPose ``` engine.init_humanface_module() ```
        engine.init_humanface_module(faceDetection=True,
                                     faceLandmark=True,
                                     advancedLandmark=True,
                                     iris=True,
                                     tongue=True,
                                     forehead=True,
                                     attribute=True,
                                     expression=True,
                                     headPose=True,
                                     useExternalDetection=True)

        image = cv2.imread("human_face_group_image_3x768x1024.jpg")
        demo_image = image.copy()

        # NOTE : BVT python biding only support PIXEL_RGB so you need to convert color by yourself
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logging.debug("image shape : %s", image.shape)

        # you need to provide external detection input
        external_input = [{
            'x': 478,
            'y': 434,
            'width': 74,
            'height': 74
        }, {
            'x': 241,
            'y': 458,
            'width': 70,
            'height': 70
        }]
        # NOTE : 0 for image mode, 1 for video mode;  Not recommended for mixed use image_mode and video_mode
        # output result is a list of detected faces, format is: [{face_0}, {face_1}, {face_2}, ..., {face_n}]
        res = engine.get_face(image, det_interval=1, run_mode=0, external_input=external_input)
        self.assertEqual(len(res), 2)

        for face_index, face in enumerate(res):
            # NOTE : face == res[face_index]
            landmark = face.landmark
            for pt in landmark:
                x = int(pt[0])
                y = int(pt[1])
                cv2.circle(demo_image, center=(x, y), radius=3, color=(255, 0, 0), thickness=-1)
        if is_viz:
            cv2.imshow("demo", demo_image)
            cv2.waitKey()

    def test_face_module_with_external_landmark(self):
        logging.debug('Face module test with external landmark')
        # NOTE : if you can't run site.getsitepackages(), you need add argument model_dir="bvt model dir"
        engine = BVT.Engine(model_dir=model_dir)

        # NOTE : by default only open faceDetection, faceLandmark, headPose ``` engine.init_humanface_module() ```
        engine.init_humanface_module(faceDetection=True,
                                     faceLandmark=True,
                                     advancedLandmark=True,
                                     iris=True,
                                     tongue=True,
                                     forehead=True,
                                     attribute=True,
                                     expression=True,
                                     headPose=True,
                                     useExternalDetection=True,
                                     useExternalLandmark=True)

        image = cv2.imread("human_face_group_image_3x768x1024.jpg")
        demo_image = image.copy()

        # NOTE : BVT python biding only support PIXEL_RGB so you need to convert color by yourself
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logging.debug("image shape : %s", image.shape)

        # you need to provide external landmark input
        external_input = [{
            'landmark': [[234.384521484375, 477.8668518066406], [234.1572265625, 483.0876159667969],
                         [234.09986877441406, 488.3321838378906], [234.30104064941406, 493.5578918457031],
                         [234.6894989013672, 498.7603759765625], [235.2947235107422, 503.8746337890625],
                         [236.1589813232422, 508.9507141113281], [237.4886474609375, 513.85546875],
                         [239.42376708984375, 518.49560546875], [242.0712890625, 522.7153930664062],
                         [245.2974853515625, 526.4965209960938], [248.83517456054688, 529.9270629882812],
                         [252.71109008789062, 533.0217895507812], [256.7253112792969, 535.8737182617188],
                         [260.7109680175781, 538.3247680664062], [264.86224365234375, 540.1566162109375],
                         [269.33551025390625, 541.440673828125], [273.9481506347656, 542.0906372070312],
                         [278.830078125, 541.6162719726562], [283.3183288574219, 539.7444458007812],
                         [287.25445556640625, 536.9349975585938], [290.8808898925781, 533.6171875],
                         [294.2010498046875, 529.9019775390625], [297.05474853515625, 525.76318359375],
                         [299.3695373535156, 521.2548217773438], [301.150146484375, 516.51318359375],
                         [302.5326843261719, 511.6703186035156], [303.7111511230469, 506.8106689453125],
                         [304.7979431152344, 501.98822021484375], [305.7668762207031, 497.1509094238281],
                         [306.4546813964844, 492.2654724121094], [306.9082946777344, 487.3513488769531],
                         [307.2348327636719, 482.4330139160156], [242.90977478027344, 471.94915771484375],
                         [248.3926239013672, 468.8341979980469], [254.4696807861328, 468.08978271484375],
                         [260.57415771484375, 469.578125], [266.4083557128906, 471.80059814453125],
                         [282.7071228027344, 473.0788879394531], [288.01348876953125, 471.480712890625],
                         [293.5028991699219, 470.599609375], [298.7720947265625, 471.8211669921875],
                         [303.1349182128906, 475.2437438964844], [274.05712890625, 482.7977600097656],
                         [273.7554931640625, 489.85443115234375], [273.4661865234375, 496.9782409667969],
                         [273.1678771972656, 504.08746337890625], [263.52490234375, 507.9824523925781],
                         [267.89068603515625, 509.1844787597656], [272.2242126464844, 510.422607421875],
                         [276.4674072265625, 509.84588623046875], [280.6431579589844, 509.04327392578125],
                         [248.7909698486328, 480.14288330078125], [252.51815795898438, 479.1231689453125],
                         [259.2124938964844, 479.50115966796875], [262.5310363769531, 481.4428405761719],
                         [258.77099609375, 482.00006103515625], [252.3321533203125, 481.50946044921875],
                         [284.1264343261719, 483.3902587890625], [287.62841796875, 481.3605651855469],
                         [294.4354248046875, 481.6550598144531], [297.5599060058594, 483.8985290527344],
                         [294.0992126464844, 485.1602783203125], [287.6895751953125, 484.7107849121094],
                         [248.84707641601562, 471.86810302734375], [254.61770629882812, 472.2268371582031],
                         [260.233154296875, 473.5387268066406], [265.69842529296875, 475.214111328125],
                         [283.0945129394531, 476.0667724609375], [287.98687744140625, 474.90911865234375],
                         [292.9837951660156, 474.12896728515625], [298.03070068359375, 474.3726501464844],
                         [255.88153076171875, 479.07635498046875], [255.54037475585938, 481.9606628417969],
                         [255.71095275878906, 480.51849365234375], [291.06396484375, 481.1096496582031],
                         [290.8868103027344, 485.1815185546875], [290.97540283203125, 483.14556884765625],
                         [268.7365417480469, 483.2172546386719], [278.6560363769531, 483.8287658691406],
                         [264.2401123046875, 499.47869873046875], [281.00909423828125, 500.63604736328125],
                         [261.087646484375, 504.2869873046875], [283.30279541015625, 505.8077087402344],
                         [255.6155548095703, 518.3099365234375], [261.59442138671875, 518.2990112304688],
                         [267.3332824707031, 518.0464477539062], [270.8042907714844, 518.7058715820312],
                         [274.2861328125, 518.5670166015625], [279.59368896484375, 519.362548828125],
                         [285.05078125, 520.1812133789062], [280.6886901855469, 525.0638427734375],
                         [275.0379638671875, 528.224853515625], [270.2707214355469, 528.5703125],
                         [265.4134521484375, 527.4533081054688], [259.8321838378906, 523.5499267578125],
                         [257.5503845214844, 519.1239013671875], [264.15301513671875, 520.5908203125],
                         [270.5999755859375, 521.7654418945312], [276.80792236328125, 521.507080078125],
                         [283.1423034667969, 520.84033203125], [277.08416748046875, 522.5453491210938],
                         [270.56610107421875, 522.9861450195312], [263.8345031738281, 521.5521850585938],
                         [255.71095275878906, 480.51849365234375], [290.97540283203125, 483.14556884765625]]
        }]
        # NOTE : 0 for image mode, 1 for video mode;  Not recommended for mixed use image_mode and video_mode
        # output result is a list of detected faces, format is: [{face_0}, {face_1}, {face_2}, ..., {face_n}]
        res = engine.get_face(image, det_interval=1, run_mode=0, external_input=external_input)
        self.assertEqual(len(res), 1)

        for face_index, face in enumerate(res):
            # NOTE : face == res[face_index]
            landmark = face.landmark
            head_pose = (face.pitch, face.yaw, face.roll)
            for pt in landmark:
                x = int(pt[0])
                y = int(pt[1])
                cv2.circle(demo_image, center=(x, y), radius=3, color=(255, 0, 0), thickness=-1)
        if is_viz:
            cv2.imshow("demo", demo_image)
            cv2.waitKey()

    def test_face_module_abtest_flag(self):
        pass



class HairSegModuleTest(unittest.TestCase):

    def test_hair_module(self):
        logging.debug('Hair Seg module test')
        # NOTE : if you can't run site.getsitepackages(), you need add argument model_dir="bvt model dir"
        engine = BVT.Engine(model_dir=model_dir)
        status = engine.init_hair_module()
        if not status:
            self.assertTrue(status)
            logging.error('Init hair segmentation module failed')
            return

        # NOTE : maybe you need a better image to test hair segmodule
        image = cv2.imread("hand_uchar_image_3x960x540.jpg")

        # NOTE : BVT python biding only support PIXEL_RGB so you need to convert color by yourself
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # NOTE : result mask shape may not equal to input image shape.Ask zhangshuye/wangyifeng for more details
        res = engine.get_hair_mask(image)
        img = res.mask
        h = res.height
        w = res.width

        mask = np.array(img, dtype=np.uint8)
        mask = np.reshape(mask, (h, w))
        if is_viz:
            cv2.imshow("demo", mask)
            cv2.waitKey()


class HeadSegTest(unittest.TestCase):

    def test_headseg(self):
        face_engine = BVT.Engine(model_dir=model_dir)
        status = face_engine.init_humanface_module(faceDetection=True,
                                                   faceLandmark=True,
                                                   advancedLandmark=True,
                                                   iris=True,
                                                   tongue=True,
                                                   forehead=True,
                                                   attribute=True,
                                                   expression=True,
                                                   headPose=True)
        if not status:
            logging.error('Init face module failed')
            self.assertTrue(status)
            return

        image = cv2.imread("human_face_group_image_3x768x1024.jpg")
        demo_image = image.copy()

        # NOTE : BVT python biding only support PIXEL_RGB so you need to convert color by yourself
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logging.debug("image shape : %s", image.shape)

        faces = face_engine.get_face(image, det_interval=1, run_mode=0, run_level="platinum")

        logging.debug(faces[0].landmark)
        seg_engine = BVT.Engine(model_dir=model_dir)
        seg_engine.init_head_module()
        res = seg_engine.get_head_mask(image, faces)
        # print(res)

        mask = np.array(res.mask, dtype=np.uint8)
        print(np.sum(mask))
        mask = np.reshape(mask, (res.height, res.width))
        if is_viz:
            cv2.imshow("demo", mask)
            cv2.waitKey()


class GetMnnResTest(unittest.TestCase):

    def test_getmnnres(self):
        image = cv2.imread("cartoon_style_transfer_popart_3x1280x720.jpg")
        demo_image = image.copy()

        # NOTE : BVT python biding only support PIXEL_RGB so you need to convert color by yourself
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logging.debug("image shape : %s", image.shape)
        engine = BVT.Engine(model_dir=model_dir)
        site_package_path = "/home/xiang/CLionProjects/BVT/python_binding_2"
        test_model_dir = os.path.join(site_package_path, "BVTData", "bvtMobile", "models")
        model_path = os.path.join(test_model_dir, "head_seg_model_quantized_fast.mnn")
        """
           getMnnRes(self, mnnPath, netInputBlobName, netOutBlobName, meanList, varList, netInputW, netInputH, bvtPixelType, imagedata, outputW, outputH, outputC):

           mnnPath：test mnn model path

           netInputBlobName

           netOutBlobName

           meanList

           valList

           notes:preprocess method:（data-mean）*var

           netInputW，netInputH：net input width, net input height

           bvtPixelType：internal pixel type conversion, index description are as follows

           0:YUV2RGB
           1:RGB2RGB
           2:BGR2BGR
           3:RGBA2RGBA
           4:GRAY2GRAY
           5:RGB2BGR
           6:BGR2RGB
           7:RGBA2RGB
           7:RGBA2BGR

           imagedata: input image data

           outputW, outputH, outputC：output width, output height, output channel
        """
        mask, w, h, c = engine.getMnnRes(model_path, "data", "deconv_upsample_3", [0.0, 0.0, 0.0],
                                         [0.00392, 0.00392, 0.00392], 128, 128, 1, image, 128, 128, 1)
        # print(res)
        print(w, h, c)

        mask = np.array(mask)
        mask = np.where(mask > 1, 1, mask)
        mask = np.where(mask < 0, 0, mask)
        mask = (255 * mask).astype(np.uint8)
        mask = mask.reshape(w, h, c)

        if is_viz:
            cv2.imshow("demo", mask)
            cv2.waitKey()


class FaceSegTest(unittest.TestCase):

    def test_face_parsing(self):
        engine = BVT.Engine(model_dir=model_dir)
        status = engine.init_face_parsing()
        if not status:
            logging.error('Init face parsing failed')
            self.assertTrue(status)
            return

        image = cv2.imread("human_face_group_image_3x768x1024.jpg")
        demo_image = image.copy()

        # NOTE : BVT python biding only support PIXEL_RGB so you need to convert color by yourself
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logging.debug("image shape : %s", image.shape)

        res = engine.get_face_parsing(image, run_mode=0, run_level="platinum")
        mask = np.array(res.mask, dtype=np.uint8)
        mask = np.reshape(mask, (res.height, res.width, 1))
        if is_viz:
            background = np.zeros_like(image)
            mask_res = demo_image.astype(float) * mask.astype(float) / 255. + background.astype(float) * (
                1 - mask.astype(float) / 255.)
            cv2.imshow("mask_res", mask_res.astype(np.uint8))
            cv2.imshow("demo", mask)
            cv2.waitKey()


class HalfBodySegModuleTest(unittest.TestCase):

    def test_half_body_module(self):
        logging.debug('Half Body Seg module test')
        # NOTE : if you can't run site.getsitepackages(), you need add argument model_dir="bvt model dir"
        engine = BVT.Engine(model_dir=model_dir)
        status = engine.init_half_body_module()
        if not status:
            self.assertTrue(status)
            logging.error('Init half body segmentation module failed')
            return

        # NOTE : maybe you need a better image to test hair segmodule
        image = cv2.imread("pose_uchar_image_3x960x540.jpg")

        # NOTE : BVT python biding only support PIXEL_RGB so you need to convert color by yourself
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # NOTE : result mask shape may not equal to input image shape.Ask zhangshuye/wangyifeng for more details
        res = engine.get_half_body_mask(image)

        mask = np.array(res.mask, dtype=np.uint8)
        mask = np.reshape(mask, (res.height, res.width))
        if is_viz:
            cv2.imshow("demo", mask)
            cv2.waitKey()


class CatFaceModuleTest(unittest.TestCase):

    def test_cat_face(self):
        logging.debug('Cat face module test')
        # NOTE : if you can't run site.getsitepackages(), you need add argument model_dir="bvt model dir"
        engine = BVT.Engine(model_dir=model_dir)
        status = engine.init_cat_module()
        if not status:
            self.assertTrue(status)
            logging.error('Init cat face module failed')
            return

        # NOTE : maybe you need a better image to test hair segmodule
        image = cv2.imread("catface.jpeg")

        # NOTE : BVT python biding only support PIXEL_RGB so you need to convert color by yourself
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # NOTE : result mask shape may not equal to input image shape.Ask zhangshuye/wangyifeng for more details
        res = engine.get_catface(image)
        for pt in res[0].landmarks:
            print(pt)
        print(res[0].bbox)
        print(res[0].id)


class HandGestureModuleTest(unittest.TestCase):

    def test_hand_module(self):
        logging.debug('Hand Gesture module test')
        # NOTE : if you can't run site.getsitepackages(), you need add argument model_dir="bvt model dir"
        engine = BVT.Engine(model_dir=model_dir)

        status = engine.init_hand_gesture_module()
        if not status:
            self.assertTrue(status)
            logging.error('Init hand gesture module failed')
            return

        image = cv2.imread("hand_uchar_image_3x960x540.jpg")

        # NOTE : BVT python biding only support PIXEL_RGB so you need to convert color by yourself
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # NOTE : hand gesture module do not have image mode and have some tracking tricks
        # NOTE : so you need to run get_hand_gesture 3 times to get detection results
        # NOTE : and you need to run another 3 times (6 times in total) to get classification results
        # TODO [{'bbox': [71.0, 474.0, 377.0, 369.0], 'point': [259.5, 658.5], 'label': 2, 'prob': 0.7042252421379089}] ——>[<BVT.PYHandOutData object at 0x7f25a2770dc0>]
        loop = 0
        while True:
            res = engine.get_hand_gesture(image)
            if len(res) < 0 and loop < 10:
                time.sleep(1)
                loop += 1
            else:
                break
        if len(res) > 0:
            logging.debug(res[0].bbox)


class CartoonTest(unittest.TestCase):

    def test_cartoon(self):
        engine = BVT.Engine(model_dir=model_dir)

        image_path = os.path.join(TEST_DATA, "cartoon_style_transfer_popart_3x1280x720.jpg")
        image = cv2.imread(image_path)
        h, w, c = image.shape

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        engine.init_cartoon()

        res = engine.run_cartoon(image)
        res = engine.run_cartoon(image)
        res = engine.run_cartoon(image)
        res = engine.run_cartoon(image)
        res = engine.run_cartoon(image)
        res = np.array(res, dtype=np.uint8)
        res = np.reshape(res, (h, w, 4))
        mask = res[:, :, 3:] / 255.0
        res = res[:, :, :3]
        image = image * (1.0 - mask) + res * mask
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if is_viz:
            cv2.imshow("res", res)
            cv2.imshow("image", image)
            cv2.waitKey()


class PoseTest(unittest.TestCase):

    def test_pose(self):
        engine = BVT.Engine(model_dir=model_dir)

        image_path = os.path.join(TEST_DATA, "pose_uchar_image_3x960x540.jpg")
        image = cv2.imread(image_path)
        h, w, c = image.shape

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        res = engine.init_human_pose_module()
        if not res:
            self.assertTrue(res)
            logging.error('Init pose module failed')
            return
        logging.debug(image.shape)
        res = engine.get_human_pose(image)

        for pt in res[0].landmark:
            # pt : [x, y, label, conf, visible]
            x, y, label, conf, visible = pt
            logging.debug(pt)
            if visible > 0:
                cv2.circle(image, center=(int(x), int(y)), color=(255, 0, 0), radius=3, thickness=-1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if is_viz:
            cv2.imshow("image", image)
            cv2.waitKey()


class CoverSelectorTest(unittest.TestCase):

    def test_cover(self):
        engine = BVT.Engine(model_dir=model_dir)

        image_path = os.path.join(TEST_DATA, "cartoon_style_transfer_popart_3x1280x720.jpg")
        image = cv2.imread(image_path)
        h, w, c = image.shape

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        engine.init_cover_selector()

        recommend_score = engine.get_recommend(image)
        softporn_score, horror_score = engine.get_moderation(image)

        logging.info("recommend_score: %f", recommend_score)
        logging.info("softporn_score: %f", softporn_score)
        logging.info("horror_score: %f", horror_score)


class IQATest(unittest.TestCase):

    def test_iqa(self):
        engine = BVT.Engine(model_dir=model_dir)

        image_path = os.path.join(TEST_DATA, "iqa_uchar_image_3x480x640.jpg")
        image = cv2.imread(image_path)
        h, w, c = image.shape

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        res = engine.init_image_quality_assessment()
        if not res:
            self.assertTrue(res)
            logging.error('Init pose module failed')
            return

        res = engine.get_image_quality(image)
        logging.info(res)
        print(res.faceQuality[0].all)


if __name__ == '__main__':
    unittest.main(verbosity=2)
