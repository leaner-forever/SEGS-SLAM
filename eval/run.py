import numpy as np
import os
import sys
import time
import torch
from tqdm import tqdm, trange
from PIL import Image
import json
import glob
from argparse import ArgumentParser, Namespace
from utils import loadPose, LPIPS, img2mse, mse2psnr
import yaml
import cv2
import torchvision.transforms as tf
import torchvision.transforms.functional as tfunc

import torch.multiprocessing as mp
from loss_utils import ssim
from image_utils import psnr
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, Dataset
import gc
from pytorch_msssim import ssim as pssim

def loadReplica(path):
    color_paths = sorted(glob.glob(os.path.join(path, "results/frame*.jpg")))
    #print(path, color_paths)
    tstamp = [float(color_path.split("/")[-1].replace("frame", "").replace(".jpg", "").replace(".png", "")) for color_path in color_paths]
    return color_paths, tstamp

def loadTUM(path):
    if os.path.exists(os.path.join(path, "rgb3")):
        color_paths = sorted(glob.glob(os.path.join(path, "rgb3/*.png")))
    else:
        color_paths = sorted(glob.glob(os.path.join(path, "rgb/*.png")))
    #print(path, color_paths)
    tstamp = [float(color_path.split("/")[-1].replace("frame", "").replace(".jpg", "").replace(".png", "")) for color_path in color_paths]
    return color_paths, tstamp

def loadKITTI(path):
    color_paths = sorted(glob.glob(os.path.join(path, "image_2/*.png")))
    #print(path, color_paths)
    tstamp = np.loadtxt(os.path.join(path, "times.txt"), delimiter=' ', dtype=np.unicode_).astype(np.float32)
    #tstamp = [float(color_path.split("/")[-1].replace("frame", "").replace(".jpg", "").replace(".png", "")) for color_path in color_paths]
    return color_paths, tstamp

def loadEuRoC(path):
    #print(path)
    color_paths = sorted(glob.glob(os.path.join(path, 'mav0/cam0/data/*.png')))    
    #print(color_paths) 
    tstamp = [np.float64(x.split('/')[-1][:-4])/1e9 for x in color_paths]
    return color_paths, tstamp

def associate_frames(tstamp_image, tstamp_pose, max_dt=0.08):
    """ pair images, depths, and poses """
    associations = []
    for i, t in enumerate(tstamp_image):
        j = np.argmin(np.abs(tstamp_pose - t))
        if (np.abs(tstamp_pose[j] - t) < max_dt):
            associations.append((i, j))
    return associations

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []

    renders_dir = Path(renders_dir)
    gt_dir = Path(gt_dir)

    file_list = os.listdir(renders_dir)
    for fname in tqdm(file_list, desc="Reading Images"):
        # print(fname)
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tfunc.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tfunc.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


if __name__ == "__main__":
    mp.set_start_method('spawn')
    # Set up command line argument parser
    parser = ArgumentParser(description="evaluation script parameters")
    parser.add_argument("result_path", type=str, default = None)
    parser.add_argument("gt_path", type=str, default = None)
    parser.add_argument("--correct_scale", action="store_true")
    parser.add_argument("--show_plot", action="store_true")
    parser.add_argument('--train', action='store_true', help="A boolean flag (default: False)")#eval train view
    args = parser.parse_args()
    dirs = os.listdir(args.result_path)
    # load model
    ts = []
    Rs = []
    render_time = 0
    for file_name in dirs:
        #print(file_name)
        if "shutdown" in file_name:
            render_time = np.loadtxt(os.path.join(args.result_path, file_name, "render_time.txt"), delimiter=' ', dtype=np.unicode_)
            render_time = render_time[:, 1].astype(np.float32)

        if "images" in file_name:
            if args.train :
                renders_dir =os.path.join(args.result_path, file_name, "kf_image") #train view
                gt_dir =os.path.join(args.result_path, file_name, "kf_image_gt")
            else :
                renders_dir =os.path.join(args.result_path, file_name, "all_image") #novel view
                gt_dir =os.path.join(args.result_path, file_name, "all_image_gt")

            image_names = []
            lpipss = []

            renders_dir = Path(renders_dir)
            gt_dir = Path(gt_dir)

            file_list = os.listdir(renders_dir)
            for fname in tqdm(file_list, desc="Reading Images"):
                with torch.no_grad():
                    render = Image.open(renders_dir / fname)
                    gt = Image.open(gt_dir / fname)
                    image_names.append(fname)
                    render_tensor = tfunc.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
                    gt_tensor = tfunc.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
                    
                    render_img = render_tensor.squeeze(0)
                    gt_img  = gt_tensor.squeeze(0)
                    val_lpips = LPIPS.calculate(render_img.type(torch.float32).permute(1, 2, 0),
                                                gt_img.type(torch.float32).permute(1, 2, 0), render_img.device)
                    lpipss.append(val_lpips)
                    val_loss = img2mse(render_img, gt_img)
                    del render_tensor
                    del gt_tensor
                    del gt_img
                    del render_img            
                    torch.cuda.empty_cache()
                    gc.collect()

            torch.cuda.empty_cache()

            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("")

            lpipss_tensor = torch.tensor(lpipss)
            lpips_list = lpipss_tensor.cpu().numpy()
            np.savetxt(os.path.join(args.result_path, "lpips.txt"), lpips_list)

    with open(os.path.join(args.result_path, "TrackingTime.txt"), "r") as fin:
        tracking_time = fin.readlines()
    tracking_time = np.array(tracking_time[:-3]).astype(np.float32)
    with open(os.path.join(args.result_path, "eval.txt"), "w") as fout:
        fout.write("lpips: {}\n".format(np.mean(lpips_list)))
        fout.write("tracking s: {}\n".format(np.mean(tracking_time)))
        fout.write("tracking FPS: {}\n".format(1/np.mean(tracking_time)))

        fout.write("rendering ms: {}\n".format(np.mean(render_time)))
        fout.write("rendering FPS: {}\n".format(1000/np.mean(render_time)))
    
    print("lpips: {}".format(np.mean(lpips_list)))
    print("tracking s: {}".format(np.mean(tracking_time)))
    print("tracking FPS: {}".format(1/np.mean(tracking_time)))

    print("rendering ms: {}".format(np.mean(render_time)))
    print("rendering FPS: {}".format(1000/np.mean(render_time)))    

    #### pose evaluation
    import evo
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation
    # load estimated poses
    pose_path = os.path.join(args.result_path, "CameraTrajectory_TUM.txt")
    traj_est = file_interface.read_tum_trajectory_file(pose_path)
    # load gt pose
    if "kitti" in args.gt_path.lower():
        def loadKITTIPose(gt_path):
            scene = gt_path.split("/")[-1]
            gt_file = gt_path.replace(scene, 'poses/{}.txt'.format(scene))
            pose_quat = []
            with open(gt_file, "r") as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    line = lines[i].split()
                    #print(line)
                    c2w = np.array(list(map(float, line))).reshape(3, 4)
                    #print(c2w)
                    quat = np.zeros(7)
                    quat[:3] = c2w[:3, 3]
                    quat[3:] = Rotation.from_matrix(c2w[:3, :3]).as_quat()
                    pose_quat.append(quat)
            pose_quat = np.array(pose_quat)

            return pose_quat
        
        pose_quat = loadKITTIPose(args.gt_path)
        traj_ref = PoseTrajectory3D(positions_xyz=pose_quat[:,:3],         
                                        orientations_quat_wxyz=pose_quat[:,3:],         
                                        timestamps=np.array(gt_tstamp))

        #scene = args.gt_path.split("/")[-1]
        #gt_file = args.gt_path.replace(scene, 'poses/{}.txt'.format(scene))
        #print(gt_file)
        #traj_ref = file_interface.read_kitti_poses_file(gt_file)
    elif "replica" in args.gt_path.lower():
        gt_file = os.path.join(args.gt_path, 'pose_TUM.txt')
        traj_ref = file_interface.read_tum_trajectory_file(gt_file)
        #if not os.path.isfile(gt_file):
    elif "euroc" in args.gt_path.lower():
        gt_file = os.path.join(args.gt_path, 'mav0/state_groundtruth_estimate0/data.csv')
        traj_ref = file_interface.read_euroc_csv_trajectory(gt_file)
    else:
        gt_file = os.path.join(args.gt_path, 'groundtruth.txt')   
        traj_ref = file_interface.read_tum_trajectory_file(gt_file)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.1)
    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=args.correct_scale)
    result_rotation_part = main_ape.ape(traj_ref, traj_est, est_name='rot', pose_relation=PoseRelation.rotation_part, 
                                        align=True, correct_scale=args.correct_scale)

    out_path=os.path.join(args.result_path, "metrics_traj.txt")
    with open(out_path, 'w') as fp:
        fp.write(result.pretty_str())
        fp.write(result_rotation_part.pretty_str())
    print(result)

    if args.show_plot:
        from evo.tools import plot
        from evo.tools.plot import PlotMode
        import matplotlib.pyplot as plt
        import copy
        traj_est_aligned = copy.deepcopy(traj_est)
        traj_est_aligned.align(traj_ref, correct_scale=True)
        fig = plt.figure()
        traj_by_label = {
            "estimate (not aligned)": traj_est,
            "estimate (aligned)": traj_est_aligned,
            "reference": traj_ref
        }
        plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
        plt.show()
