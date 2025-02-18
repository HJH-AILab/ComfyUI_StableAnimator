import numpy as np
import torch
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from insightface.app import FaceAnalysis
import cv2
import argparse
import os

def get_face_masks(image_path, save_path, app, face_helper, height=904, width=512):
    image_1 = cv2.imread(image_path)
    height, width = image_1.shape[:2]
    image_bgr_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2BGR)
    image_info_1 = app.get(image_bgr_1)

    mask_1 = np.zeros((height, width), dtype=np.uint8)
    if len(image_info_1) > 0:
        print("This is FaceAnalysis")
        for info in image_info_1:
            x_1 = info['bbox'][0]
            y_1 = info['bbox'][1]
            x_2 = info['bbox'][2]
            y_2 = info['bbox'][3]
            cv2.rectangle(mask_1, (int(x_1), int(y_1)), (int(x_2), int(y_2)), (255), thickness=cv2.FILLED)
        cv2.imwrite(save_path, mask_1)
    else:
        face_helper.clean_all()
        with torch.no_grad():
            bboxes = face_helper.face_det.detect_faces(image_bgr_1, 0.97)
        if len(bboxes) > 0:
            print("This is FaceRestoreHelper")
            for bbox in bboxes:
                cv2.rectangle(mask_1, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255), thickness=cv2.FILLED)
            cv2.imwrite(save_path, mask_1)
        else:
            print("This is no detected face")
            mask_1[:] = 255
            cv2.imwrite(save_path, mask_1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Human Face Mask Extraction", add_help=True)
    parser.add_argument("--image_folder", type=str, help="Specify a path of a image folder")
    args = parser.parse_args()

    image_folder = args.image_folder

    app = FaceAnalysis(
        name='antelopev2', root='.', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    face_helper = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        device="cuda",
    )
    face_helper.face_parse = init_parsing_model(model_name='bisenet', device="cuda")

    print(f"images subfolder path: {image_folder}")
    face_subfolder_path = os.path.join(os.path.dirname(image_folder), "faces")
    if not os.path.exists(face_subfolder_path):
        os.makedirs(face_subfolder_path)
        print(f"Folder created: {face_subfolder_path}")
    else:
        print(f"Folder already exists: {face_subfolder_path}")
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                print(file_path)
                file_name = os.path.splitext(file)[0]
                image_name = file_name + '.png'
                image_legal_path = os.path.join(image_folder, image_name)
                if os.path.exists(os.path.join(face_subfolder_path, file_name + '.png')):
                    existed_path = os.path.join(face_subfolder_path, file_name + '.png')
                    print(f"{existed_path} already exists!")
                    continue

                face_save_path = os.path.join(face_subfolder_path, file_name + '.png')
                get_face_masks(image_path=image_legal_path, save_path=face_save_path, app=app, face_helper=face_helper)
                print(f"Finish face Extraction: {face_save_path}")