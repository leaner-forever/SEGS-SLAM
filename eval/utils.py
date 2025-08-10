import os
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation
from skimage.metrics import structural_similarity as calculate_ssim
import lpips
import torch

img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x, rank: -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(rank))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
class LPIPS:
    loss_fn_alex = None

    @staticmethod
    def calculate(img_a, img_b, rank):
        img_a, img_b = [img.permute([2, 1, 0]).unsqueeze(0) for img in [img_a, img_b]]
        if LPIPS.loss_fn_alex == None:  # lazy init
            LPIPS.loss_fn_alex = lpips.LPIPS(net='alex', version='0.1').to(rank)
        return LPIPS.loss_fn_alex(img_a.to(rank), img_b.to(rank))

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = np.tan((fovY / 2))
    tanHalfFovX = np.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

class MiniCam:
    
    def __init__(self, width, height, fovx, fovy, world_view_transform):
        self.image_width = width
        self.image_height = height  
        self.FoVx = fovx
        self.FoVy = fovy
        self.znear = 0.01
        self.zfar = 100.0
        #self.world_view_transform = world_view_transform
        #self.full_proj_transform = full_proj_transform
        #view_inv = torch.inverse(self.world_view_transform)
        #self.camera_center = view_inv[3][:3]
        self.world_view_transform = torch.tensor(world_view_transform, dtype=torch.float32).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def loadPose(path):
    poses = []
    """ 
    with open(path, "r") as fin:
        lines = fin.readlines()
    for line in lines:
        line = np.array(list(map(float, line.split())))
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(line[4:]).as_matrix()
        pose[:3, 3] = line[1:4]
        poses.append(pose)
    """
    pose_data = np.loadtxt(path, delimiter=' ', dtype=np.unicode_)
    pose_vecs = pose_data[:, 1:].astype(np.float32)
    tstamp = pose_data[:, 0].astype(np.float64)
    
    for pose_vec in pose_vecs:
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pose_vec[3:]).as_matrix()
        pose[:3, 3] = pose_vec[:3]
        poses.append(pose)
        #print(pose);exit()
    return poses, tstamp

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def focal2fov(focal, pixels):
    return 2*np.arctan(pixels/(2*focal))



