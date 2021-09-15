import argparse
import numpy as np
import cv2
import face_alignment
import os
from utils import img_scaler

def estimate_pose(img_path, fa, draw_landmark=True):
    fname = os.path.basename(img_path)
    img = cv2.imread(img_path)[..., ::-1]  # BGR to RGB
    preds_list = fa.get_landmarks(img)
    if preds_list is None:
        print(img_path, 'is passed on First Face Detection.')
        return None

    face_imgs = []
    for preds in preds_list:
        # Cut face_imgs
        min_x, max_x = round(min(preds[:, 0])), round(max(preds[:, 0]))
        min_y, max_y = round(min(preds[:, 1])), round(max(preds[:, 1]))
        min_x, max_x, min_y, max_y = tuple(map(round, img_scaler(img.shape, min_x, max_x, min_y, max_y, scale=0.35)))
        face_img = img[min_y:max_y, min_x:max_x]

        # Calculate align degree
        left_eye = preds[36:42]
        right_eye = preds[42:48]
        center_left_eye, center_right_eye = np.mean(left_eye, axis=0), np.mean(right_eye, axis=0)
        dY = center_right_eye[1] - center_left_eye[1]
        dX = center_right_eye[0] - center_left_eye[0]
        angle_rad = np.arctan2(dY, dX)
        angle = np.degrees(np.arctan2(dY, dX))

        # Face align
        face_center = round((max_x-min_x)/2) , round((max_y-min_y)/2)
        rot = cv2.getRotationMatrix2D(face_center, angle, 1)
        rot_face_img = cv2.warpAffine(face_img, rot, (0, 0))

        face_imgs.append(rot_face_img)

    results = list()
    # For Detected Faces
    for face_img in face_imgs:
        # Detect again for accuracy of landmarks
        preds_list = fa.get_landmarks(face_img)
        if preds_list is None:
            print(img_path, 'is passed on Second Face Detection.')
            return None

        # For preds_list
        for preds in preds_list:
            # Pass side face
            if np.unique(preds<0).any() == np.True_:
                continue
            
            drawed = face_img.copy()
            # Draw landmarks
            if draw_landmark is True:
                left_eye = preds[36:42]
                right_eye = preds[42:48]
                center_left_eye, center_right_eye = np.mean(left_eye, axis=0), np.mean(right_eye, axis=0)
                rad2 = 1
                cv2.circle(drawed, list(map(round, center_left_eye)), rad2, (0, 0, 255), -1)
                cv2.circle(drawed, list(map(round, center_right_eye)), rad2, (0, 0, 255), -1)
                for i, pred in enumerate(preds):
                    color = (0, 255, 0)
                    rad = 1
                    if (i+1) == 31:
                        color = (255, 0, 0)
                        rad = rad2
                    cv2.circle(drawed, list(map(round, pred)), rad, color, -1)

            # Estimate yaw & pitch ratio
            min_x, max_x = round(min(preds[:, 0])), round(max(preds[:, 0]))
            min_y, max_y = round(min(preds[:, 1])), round(max(preds[:, 1]))
            w, h = abs(min_x-max_x), abs(min_y-max_y)
            yaw_ratio = (preds[30][0]-min_x)/w*100 - 50
            pitch_ratio = (preds[30][1]-min_y)/h*100 - 50

            result = [yaw_ratio, pitch_ratio, drawed[min_y:max_y, min_x:max_x]]
            results.append(result)

    return results, img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate face yaw & pitch ratio')
    parser.add_argument('--path', required=True, help='image path')
    parser.add_argument('--device', required=False, default='cuda', help="device select 'cuda' or 'cpu'")
    parser.add_argument('--fa', required=False, default='sfd', help="select face detector 'sfd' or 'blazeface' or 'dlib'")
    parser.add_argument('--fa_thres', required=False, default=0.93, help="face detect threshold")
    parser.add_argument('--view', required=False, default=False, help='show detected face img', action='store_true')
    args = parser.parse_args()

    print('Loading Detector...', end='')
    face_detector = args.fa
    face_detector_kwargs = {
        "filter_threshold" : float(args.fa_thres)
    }
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=args.device, flip_input=False,
                                    face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)
    print('Done.')
    
    print('Estimating...', end='')
    results, _ = estimate_pose(args.path, fa, draw_landmark=args.view)
    print('Done.')
    if results is None:
        raise Exception('Face not detected!')
    for i, (yaw_ratio, pitch_ratio, face_img) in enumerate(results):
        print(f'[Face {i} - yaw_ratio: {yaw_ratio:.3f}, pitch_ratio: {pitch_ratio:.3f}]')
        if args.view:
            cv2.imshow('test_img', face_img[..., ::-1])
    if args.view:
        cv2.waitKey(0)
        cv2.destroyAllWindows()