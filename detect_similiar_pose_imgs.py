import cv2
import face_alignment
import matplotlib.pyplot as plt
import os
import argparse
from os.path import join
from tqdm import tqdm
from estimate_pose import estimate_pose
from glob import glob

def select_reference_value(img_path, fa):
    results, _ = estimate_pose(img_path, fa, draw_landmark=True)
    for i, (yaw_ratio, pitch_ratio, face_img) in enumerate(results):
        print(f'[Face {i} - yaw_ratio: {yaw_ratio:.3f}, pitch_ratio: {pitch_ratio:.3f}]')
        cv2.imshow(f'Face {i}', face_img[..., ::-1])
    print('Please press any key.')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    selected_face_num = int(input('Select a face number : '))

    return results[selected_face_num][:2]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect similiar head pose images')
    parser.add_argument('--ref', required=True, help='reference image path')
    parser.add_argument('--dpath', required=True, help='folder path to find similar head pose images')
    parser.add_argument('--save_path', required=True, default='out', help='save_folder')
    parser.add_argument('--save_debug', required=False, default=False, help='whether to detect unsimiliar imgs', action='store_true')
    parser.add_argument('--allow_yaw_flip', required=False, default=False, help='whether to allow yaw flipped imgs', action='store_true')
    parser.add_argument('--device', required=False, default='cuda', help="device select 'cuda' or 'cpu'")
    parser.add_argument('--fa', required=False, default='sfd', help="select face detector 'sfd' or 'blazeface' or 'dlib'")
    parser.add_argument('--fa_thres', required=False, default=0.93, help="face detect threshold")
    parser.add_argument('--threshold', required=False, default=7, help='yaw & pitch ratio threshold(%)')
    args = parser.parse_args()

    os.mkdir(args.save_path)
    os.mkdir(join(args.save_path, 'detected'))
    os.mkdir(join(args.save_path, 'detected_debug'))

    print('Loading Detector...', end='')
    face_detector = args.fa
    face_detector_kwargs = {
        "filter_threshold" : float(args.fa_thres)
    }
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=args.device, flip_input=False,
                                    face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)
    print('Done.')

    ref_yaw, ref_pitch = select_reference_value(args.ref, fa)

    img_list = glob(join(args.dpath, '*.??g'))
    plt.figure(figsize=(10,10))
    for img_path in tqdm(img_list):
        fname = os.path.basename(img_path).split('.')[0]
        results, img = estimate_pose(img_path, fa, draw_landmark=True)
        if results is None:
            continue
        for i, (yaw_ratio, pitch_ratio, face_img) in enumerate(results):
            if args.allow_yaw_flip:
                yaw_ratio, ref_yaw = abs(ref_yaw), abs(ref_yaw)
            save_name = f'{fname}_face_{i:05}.jpg'
            plt.clf()
            plt.title(f'yaw_ratio: {yaw_ratio:.3f}, pitch_ratio: {pitch_ratio:.3f}', fontsize=25)
            plt.imshow(face_img)
            if abs(yaw_ratio-ref_yaw) <= float(args.threshold) and abs(pitch_ratio-ref_pitch) <= float(args.threshold):
                plt.savefig(join(args.save_path, 'detected', save_name), dpi=60)
            elif args.save_debug:
                plt.savefig(join(args.save_path, 'detected_debug', save_name), dpi=60)
