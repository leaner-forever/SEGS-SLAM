import numpy as np
import os
import glob
import argparse
"""
This is a scipt to calcuate quantitative results stored in the result_main_folder for the evluation dataset
Outputs are two files including "result_main_folder/log.txt" and "result_main_folder/log.csv"

if --train, the scipt will eval the train view. Otherwise, it will eval the novel view.

python onekey.py --dataset_center_path PATH_TO_ALL_DATASET --result_main_folder /home/lzy/workingspace/Scaffold-GS-cpp/results
python onekey.py --dataset_center_path "/home/lzy/workingspace/MonoGS/datasets/" --result_main_folder "/home/lzy/workingspace/Scaffold-GS-cpp/results/" --train
"""

# 过滤并转换数据
def filter_and_convert(data):
    return [float(x) for x in data if isinstance(x, (int, float, str)) and x != 'None' and x != None]

parser = argparse.ArgumentParser(description="evaluation script")
parser.add_argument("-d", "--dataset_center_path", type=str, required=True)
parser.add_argument("-r", "--result_main_folder", type=str, required=True)
parser.add_argument('--train', action='store_true', help="A boolean flag (default: False)")#eval train view
args = parser.parse_args()

# dataset_center_path = "/homes/huajian/Dataset"
# result_main_folder = os.path.join("../result/4090/results/")

dataset_center_path = args.dataset_center_path
result_main_folder = args.result_main_folder

print(dataset_center_path)
print(result_main_folder)

gt_dataset = {"replica": {"path": os.path.join(dataset_center_path, "replica"),
                           "scenes": ['room0', 'room1', 'room2', 'office0', 'office1', 'office2', 'office3', 'office4']}, 
            "tum": {"path": os.path.join(dataset_center_path, "tum"),
                    "scenes": ['rgbd_dataset_freiburg1_desk', 'rgbd_dataset_freiburg2_xyz', 'rgbd_dataset_freiburg3_long_office_household']},
            "eth3d": {"path": os.path.join(dataset_center_path, "ETH3D"),
                        "scenes": ["desk_3", "mannequin_1", "mannequin_3", "planar_2", "planar_3", "table_7"]},
            "kitti": {"path": os.path.join(dataset_center_path, "KITTI"),
                        "scenes": ["00", "01","02","03","04","05","06","07","08","09","10"]},
            "euroc": {"path": os.path.join(dataset_center_path, "EuRoC"),
                      "scenes": ["MH_01_easy", "MH_02_easy","V1_01_easy","V2_01_easy"]}
            }

# path the all results
results = [m for m in sorted(os.listdir(result_main_folder)) if os.path.isdir(os.path.join(result_main_folder, m))]
print(results)

for result in results:
    print("processing", result)
    # support datasetName_cameratype_xx
    gt_dataset_name = result.split("_")[0].lower()
    gt_dataset_path = gt_dataset[gt_dataset_name]["path"]
    gt_dataset_scenes = gt_dataset[gt_dataset_name]["scenes"]
    print(gt_dataset_name)
    print(gt_dataset_path)
    # print(gt_dataset_path)
    for scene in gt_dataset_scenes:
        result_path = os.path.join(result_main_folder, result, scene)
        gt_path = os.path.join(gt_dataset_path, scene)
        if not os.path.exists(os.path.join(result_path, "eval.txt")):
            if "mono" in result.lower():
                if args.train :
                    os.system("python run.py {} {} --correct_scale --train".format(result_path, gt_path))
                else :
                    os.system("python run.py {} {} --correct_scale".format(result_path, gt_path))
            else:
                if args.train :
                    os.system("python run.py {} {} --train".format(result_path, gt_path))
                else :
                    os.system("python run.py {} {}".format(result_path, gt_path))


logs = [] 
camera_type = ['mono', 'rgbd', 'stereo']
#### get the result file ####
for gt_dataset_name in gt_dataset:
    # mono
    scenes = gt_dataset[gt_dataset_name]["scenes"]
    for camera in camera_type:
        results = sorted(glob.glob(os.path.join(result_main_folder, "{}_{}*".format(gt_dataset_name, camera))))
        for result in results:
            print(result)             
            logs.append(result+"\n")
            for scene in scenes:
                # T	R PSNR SSIM	LPIPS Tracking speed Rendering speed
                T, R, T_std = None, None, None
                if os.path.exists(os.path.join(result, scene, "metrics_traj.txt")):
                    with open(os.path.join(result, scene, "metrics_traj.txt")) as fin:
                        lines = fin.readlines()
                        ape_T = lines[7].split()
                        assert ape_T[0] == "rmse", result
                        T = ape_T[-1]
                        T_std = lines[9].split()[-1]
                        ape_R = lines[17].split()
                        assert ape_R[0] == "rmse", result
                        R = ape_R[-1]
                PSNR, SSIM, LPIPS, Tracking_fps, Rendering_fps =  None, None, None, None, None
                if os.path.exists(os.path.join(result, scene, "eval.txt")):    
                    with open(os.path.join(result, scene, "eval.txt")) as fin:
                        LPIPS = fin.readline().split()[-1]
                        Tracking_time = fin.readline().split()[-1]
                        Tracking_fps = fin.readline().split()[-1]
                        Rendering_time = fin.readline().split()[-1]
                        Rendering_fps = fin.readline().split()[-1]

                if os.path.exists(os.path.join(result, scene, "eval_metric.txt")):    
                    with open(os.path.join(result, scene, "eval_metric.txt")) as fin:
                        PSNR = fin.readline().split()[-1]
                        SSIM = fin.readline().split()[-1]
                
                render_path = glob.glob(os.path.join(result, scene, "*shutdown", "render_time.txt"))
                if len(render_path)>0:
                   render_time = np.loadtxt(render_path[0], delimiter=' ', dtype=np.unicode_)
                   render_time = render_time[:, 1].astype(np.float32)
                   Rendering_fps = 1000/np.mean(render_time)

                result_str = "{} {} {} {} {} {} {} {} {}\n".format(scene, T, R, PSNR, SSIM, LPIPS, Tracking_fps, Rendering_fps, T_std)
                print(result_str)
                logs.append(result_str)

print("Calculate AVG")
logs.append("AVG\n")
for gt_dataset_name in gt_dataset:
    # mono
    scenes = gt_dataset[gt_dataset_name]["scenes"]
    for camera in camera_type:
        results = sorted(glob.glob(os.path.join(result_main_folder, "{}_{}*".format(gt_dataset_name, camera))))
        if len(results) == 0:
            continue
        else :
            logs.append(gt_dataset_name+"\n")
        T_sum, R_sum, PSNR_sum, SSIM_sum, LPIPS_sum, Tracking_fps_sum, Rendering_fps_sum, T_std_sum = [], [], [], [], [], [], [], []

        for scene in scenes:
            T_avg, R_avg, PSNR_avg, SSIM_avg, LPIPS_avg, Tracking_fps_avg, Rendering_fps_avg, T_std_avg = [], [], [], [], [], [], [], []
            for result in results:
                # T	R PSNR SSIM	LPIPS Tracking speed Rendering speed
                T, R, T_std = None, None, None
                if os.path.exists(os.path.join(result, scene, "metrics_traj.txt")):
                    with open(os.path.join(result, scene, "metrics_traj.txt")) as fin:
                        lines = fin.readlines()
                        ape_T = lines[7].split()
                        assert ape_T[0] == "rmse", result
                        T = ape_T[-1]
                        T_std = lines[9].split()[-1]
                        ape_R = lines[17].split()
                        assert ape_R[0] == "rmse", result
                        R = ape_R[-1]
                PSNR, SSIM, LPIPS, Tracking_fps, Rendering_fps =  None, None, None, None, None
                if os.path.exists(os.path.join(result, scene, "eval.txt")):    
                    with open(os.path.join(result, scene, "eval.txt")) as fin:
                        # PSNR = fin.readline().split()[-1]
                        # SSIM = fin.readline().split()[-1]
                        LPIPS = fin.readline().split()[-1]
                        Tracking_time = fin.readline().split()[-1]
                        Tracking_fps = fin.readline().split()[-1]
                        Rendering_time = fin.readline().split()[-1]
                        Rendering_fps = fin.readline().split()[-1]

                if os.path.exists(os.path.join(result, scene, "eval_metric.txt")):    
                    with open(os.path.join(result, scene, "eval_metric.txt")) as fin:
                        PSNR = fin.readline().split()[-1]
                        SSIM = fin.readline().split()[-1]
                
                render_path = glob.glob(os.path.join(result, scene, "*shutdown", "render_time.txt"))
                if len(render_path)>0:
                   render_time = np.loadtxt(render_path[0], delimiter=' ', dtype=np.unicode_)
                   render_time = render_time[:, 1].astype(np.float32)
                   Rendering_fps = 1000/np.mean(render_time)

                T_avg.append(T)
                R_avg.append(R)
                PSNR_avg.append(PSNR)
                SSIM_avg.append(SSIM)
                LPIPS_avg.append(LPIPS)
                Tracking_fps_avg.append(Tracking_fps)
                Rendering_fps_avg.append(Rendering_fps)
                T_std_avg.append(T_std)

            if len(results) == 0:
                continue
            else :
                T_avg = filter_and_convert(T_avg)
                R_avg = filter_and_convert(R_avg)
                PSNR_avg = filter_and_convert(PSNR_avg)
                SSIM_avg = filter_and_convert(SSIM_avg)
                LPIPS_avg = filter_and_convert(LPIPS_avg)
                Tracking_fps_avg = filter_and_convert(Tracking_fps_avg)
                Rendering_fps_avg = filter_and_convert(Rendering_fps_avg)
                T_std_avg = filter_and_convert(T_std_avg)

                result_str = "{} {} {} {} {} {} {} {} {}\n".format(scene, np.mean(T_avg), np.mean(R_avg), np.mean(PSNR_avg), np.mean(SSIM_avg), np.mean(LPIPS_avg), np.mean(Tracking_fps_avg), np.mean(Rendering_fps_avg), np.mean(T_std_avg))
                print(result_str)
                logs.append(result_str)

                T_sum.append(np.mean(T_avg))
                R_sum.append(np.mean(R_avg))
                PSNR_sum.append(np.mean(PSNR_avg))
                SSIM_sum.append(np.mean(SSIM_avg))
                LPIPS_sum.append(np.mean(LPIPS_avg))
                Tracking_fps_sum.append(np.mean(Tracking_fps_avg))
                Rendering_fps_sum.append(np.mean(Rendering_fps_avg))
                T_std_sum.append(np.mean(T_std_avg))

        if len(results) == 0:
            continue
        else :
            result_str = "{} {} {} {} {} {} {} {} {}\n".format(gt_dataset_name+camera, np.mean(T_sum), np.mean(R_sum), np.mean(PSNR_sum), np.mean(SSIM_sum), np.mean(LPIPS_sum), np.mean(Tracking_fps_sum), np.mean(Rendering_fps_sum), np.mean(T_std_sum))
            print(result_str)
            logs.append(result_str)

#rgbd
with open(os.path.join(result_main_folder,'log.txt'), 'w') as out_file:         
    for log in logs:   
        out_file.write(log)
import csv
with open(os.path.join(result_main_folder,'log.csv'), 'w') as out_file:      
    writer = csv.writer(out_file)    
    writer.writerow(('scene', 'T', 'R', 'PSNR', 'SSIM', 'LPIPS', 'Tracking FPS', 'Rendering FPS',"T_std"))     
    for log in logs:   
        writer.writerow(log.split())


""" 
path = 
scenes =[dir for dir in os.listdir(path) if os.path.exists(os.path.join(path, dir, "traj.txt"))]
print(scenes)   

for scene in scenes:
    print("processing", scene)
    gt_path = os.path.join(path, scene)

    result_path = os.path.join("result/3080ti/replica_mono", scene)
    if not os.path.exists(os.path.join(result_path, "eval.txt")):
        os.system("python run.py {} {} --correct_scale".format(result_path, gt_path))

    result_path = os.path.join("result/3080ti/replica_mono_final", scene)
    if not os.path.exists(os.path.join(result_path, "eval.txt")):
        os.system("python run.py {} {} --correct_scale".format(result_path, gt_path))

    result_path = os.path.join("result/3080ti/replica_rgbd", scene)
    if not os.path.exists(os.path.join(result_path, "eval.txt")):
        os.system("python run.py {} {}".format(result_path, gt_path))


path = "/homes/huajian/Dataset/TUM" 
scenes =[dir for dir in os.listdir(path) if os.path.exists(os.path.join(path, dir, "rgb.txt"))] 
print(scenes) 
for scene in scenes:     
    print("processing", scene)     
    gt_path = os.path.join(path, scene)
    result_path = os.path.join("result/3080ti/tum_mono-3", scene)
    if not os.path.exists(os.path.join(result_path, "eval.txt")):
        os.system("python run.py {} {} --correct_scale".format(result_path, gt_path))

    result_path = os.path.join("result/3080ti/tum_rgbd-2", scene)
    if not os.path.exists(os.path.join(result_path, "eval.txt")):
        os.system("python run.py {} {}".format(result_path, gt_path))
"""
""" 
path = "../../dataset/ETH3D" 
scenes =[dir for dir in os.listdir(path) 
if os.path.exists(os.path.join(path, dir, "rgb.txt"))] 
print(scenes) 
for scene in scenes:     
    print("processing", scene)     
    input_folder = os.path.join(path, scene)   

"""  

