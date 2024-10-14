import os
from pathlib import Path
from tempfile import TemporaryDirectory

import librosa as lr
import matplotlib.animation as animation
import matplotlib.pyplot as plt 
import numpy as np
import soundfile as sf
import torch
from matplotlib import cm
from matplotlib.colors import ListedColormap
#from pytorch3d.transforms import (axis_angle_to_quaternion, quaternion_apply,
#                                  quaternion_multiply)
from tqdm import tqdm

from dataset.preprocess import Normalizer, vectorize_many
from dataset.quaternion import ax_to_6v

from pytorch3d.transforms import (axis_angle_to_quaternion, quaternion_apply,
                                  quaternion_multiply, quaternion_to_axis_angle, RotateAxisAngle)

from scipy.signal import butter, filtfilt
smpl_joints = [
    "root",  # 0
    "lhip",  # 1
    "rhip",  # 2
    "belly", # 3
    "lknee", # 4
    "rknee", # 5
    "spine", # 6
    "lankle",# 7
    "rankle",# 8
    "chest", # 9
    "ltoes", # 10
    "rtoes", # 11
    "neck",  # 12
    "linshoulder", # 13
    "rinshoulder", # 14
    "head", # 15
    "lshoulder", # 16
    "rshoulder",  # 17
    "lelbow", # 18
    "relbow",  # 19
    "lwrist", # 20 (60, 61, 62)
    "rwrist", # 21 (63, 64, 65)
    "lhand", # 22
    "rhand", # 23
]

smpl_parents = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
]

smpl_offsets = [
    [0.0, 0.0, 0.0],
    [0.05858135, -0.08228004, -0.01766408],
    [-0.06030973, -0.09051332, -0.01354254],
    [0.00443945, 0.12440352, -0.03838522],
    [0.04345142, -0.38646945, 0.008037],
    [-0.04325663, -0.38368791, -0.00484304],
    [0.00448844, 0.1379564, 0.02682033],
    [-0.01479032, -0.42687458, -0.037428],
    [0.01905555, -0.4200455, -0.03456167],
    [-0.00226458, 0.05603239, 0.00285505],
    [0.04105436, -0.06028581, 0.12204243],
    [-0.03483987, -0.06210566, 0.13032329],
    [-0.0133902, 0.21163553, -0.03346758],
    [0.07170245, 0.11399969, -0.01889817],
    [-0.08295366, 0.11247234, -0.02370739],
    [0.01011321, 0.08893734, 0.05040987],
    [0.12292141, 0.04520509, -0.019046],
    [-0.11322832, 0.04685326, -0.00847207],
    [0.2553319, -0.01564902, -0.02294649],
    [-0.26012748, -0.01436928, -0.03126873],
    [0.26570925, 0.01269811, -0.00737473],
    [-0.26910836, 0.00679372, -0.00602676],
    [0.08669055, -0.01063603, -0.01559429],
    [-0.0887537, -0.00865157, -0.01010708],
]

def set_line_data_3d(line, x):
    line.set_data(x[:, :2].T)
    line.set_3d_properties(x[:, 2])


def set_scatter_data_3d(scat, x, c):
    scat.set_offsets(x[:, :2])
    scat.set_3d_properties(x[:, 2], "z")
    scat.set_facecolors([c])


def get_axrange(poses):
    pose = poses[0]
    x_min = pose[:, 0].min()
    x_max = pose[:, 0].max()

    y_min = pose[:, 1].min()
    y_max = pose[:, 1].max()

    z_min = pose[:, 2].min()
    z_max = pose[:, 2].max()

    xdiff = x_max - x_min
    ydiff = y_max - y_min 
    zdiff = z_max - z_min

    biggestdiff = max([xdiff, ydiff, zdiff])
    return biggestdiff


def plot_single_pose(num, poses, lines, ax, axrange, scat, contact):
    pose = poses[num]
#    static = contact[num]
    indices = [7, 8, 10, 11]

    for i, (point, idx) in enumerate(zip(scat, indices)):
        position = pose[idx : idx + 1]
#        color = "r" if static[i] else "g"
        color = "r"
        set_scatter_data_3d(point, position, color)

    for i, (p, line) in enumerate(zip(smpl_parents, lines)):
        # don't plot root
        if i == 0:
            continue
        # stack to create a line
        data = np.stack((pose[i], pose[p]), axis=0)
        set_line_data_3d(line, data)

    if num == 0:
        if isinstance(axrange, int):
            axrange = (axrange, axrange, axrange)
        xcenter, ycenter, zcenter = 0, 0, 2.5
        stepx, stepy, stepz = axrange[0] / 2, axrange[1] / 2, axrange[2] / 2

        x_min, x_max = xcenter - stepx, xcenter + stepx
        y_min, y_max = ycenter - stepy, ycenter + stepy
        z_min, z_max = zcenter - stepz, zcenter + stepz

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)


def skeleton_render(
    poses,
    epoch=0,
    out="renders",
    name="",
    sound=True,
    stitch=False,
    sound_folder="ood_sliced",
    contact=None,
    render=True
):
    if render:
        # generate the pose with FK
        Path(out).mkdir(parents=True, exist_ok=True)
        num_steps = poses.shape[0]
        
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        
        point = np.array([0, 0, 1])
        normal = np.array([0, 0, 1])
        d = -point.dot(normal)
        xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 2), np.linspace(-1.5, 1.5, 2))
        z = (-normal[0] * xx - normal[1] * yy - d) * 1.0 / normal[2]
        # plot the plane
        ax.plot_surface(xx, yy, z, zorder=-11, cmap=cm.twilight)
        # Create lines initially without data
        lines = [
            ax.plot([], [], [], zorder=10, linewidth=1.5)[0]
            for _ in smpl_parents
        ]
        scat = [
            ax.scatter([], [], [], zorder=10, s=0, cmap=ListedColormap(["r", "g", "b"]))
            for _ in range(4)
        ]
        axrange = 3

        # create contact labels
#        feet = poses[:, (7, 8, 10, 11)]
#        feetv = np.zeros(feet.shape[:2])
#        feetv[:-1] = np.linalg.norm(feet[1:] - feet[:-1], axis=-1)
#        if contact is None:
#            contact = feetv < 0.01
#        else:
#            contact = contact > 0.95

        # Creating the Animation object
        anim = animation.FuncAnimation(
            fig,
            plot_single_pose,
            num_steps,
            fargs=(poses, lines, ax, axrange, scat, contact),
            interval=1000 // 30,
        )
    if sound:
        # make a temporary directory to save the intermediate gif in
        if render:
            temp_dir = TemporaryDirectory()
            gifname = os.path.join(temp_dir.name, f"{epoch}.gif")
            anim.save(gifname)

        # stitch wavs
        if stitch:
            assert type(name) == list  # must be a list of names to do stitching
            name_ = [os.path.splitext(x)[0] + ".wav" for x in name]
            audio, sr = lr.load(name_[0], sr=None)
            ll, half = len(audio), len(audio) // 2
            total_wav = np.zeros(ll + half * (len(name_) - 1))
            total_wav[:ll] = audio
            idx = ll
            for n_ in name_[1:]:
                audio, sr = lr.load(n_, sr=None)
                total_wav[idx : idx + half] = audio[half:]
                idx += half
            # save a dummy spliced audio
            audioname = f"{temp_dir.name}/tempsound.wav" if render else os.path.join(out, f'{epoch}_{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.wav')
            sf.write(audioname, total_wav, sr)
            outname = os.path.join(
                out,
                f'{epoch}_{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.mp4',
            )
        else:
            assert type(name) == str
            assert name != "", "Must provide an audio filename"
            audioname = name
            outname = os.path.join(
                out, f"{epoch}_{os.path.splitext(os.path.basename(name))[0]}.mp4"
            )
        if render:
            out = os.system(
                f"ffmpeg -loglevel error -stream_loop 0 -y -i {gifname} -i {audioname} -shortest -c:v libx264 -crf 26 -c:a aac -q:a 4 {outname}"
            )
    else:
        if render:
            # actually save the gif
            path = os.path.normpath(name)
            pathparts = path.split(os.sep)
            gifname = os.path.join(out, f"{pathparts[-1][:-4]}.gif")
            anim.save(gifname, savefig_kwargs={"transparent": True, "facecolor": "none"},)
    plt.close()


class SMPLSkeleton:
    def __init__(
        self, device=None,
    ):
        offsets = smpl_offsets
        parents = smpl_parents
        assert len(offsets) == len(parents)

        self._offsets = torch.Tensor(offsets).to(device)
        self._parents = np.array(parents)
        self._compute_metadata()

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)

    def forward(self, rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 3) tensor of axis-angle rotations describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        assert len(rotations.shape) == 4
        assert len(root_positions.shape) == 3
        # transform from axis angle to quaternion
        rotations = axis_angle_to_quaternion(rotations)

        positions_world = []
        rotations_world = []

        expanded_offsets = self._offsets.expand(
            rotations.shape[0],
            rotations.shape[1],
            self._offsets.shape[0],
            self._offsets.shape[1],
        )

        # Parallelize along the batch and time dimensions
        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0]) #Root is its own local and global rotation
            else:
                positions_world.append(
                    quaternion_apply(
                        rotations_world[self._parents[i]], expanded_offsets[:, :, i]
                    )
                    + positions_world[self._parents[i]]
                )
                if self._has_children[i]:
                    #User code: convert quaternion to axis_angles
                    global_rot = quaternion_multiply(rotations_world[self._parents[i]], rotations[:, :, i]) #Find child's rotation
                    rotations_world.append(global_rot)
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    # Yes, it's useless for position calculation, but I still wanna know its global orientation
#                    print("Final")
#                    rotations_world.append(None)
                    global_rot = quaternion_multiply(rotations_world[self._parents[i]], rotations[:, :, i]) #Find child's rotation
#                    global_rot = quaternion_to_axis_angle(global_rot)
                    rotations_world.append(global_rot)

        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2), rotations_world

def smplToPosition(pos, q, scale, aist = True):
    smpl = SMPLSkeleton()
    # to Tensor
    pos /= scale #Normalize by scale
    root_pos = torch.Tensor(np.array([pos]))
    local_q = torch.Tensor(np.array([q]))
    # to ax
    bs, sq, c = local_q.shape
    local_q = local_q.reshape((bs, sq, -1, 3))
    if aist:
        # AISTPP dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
        root_q = local_q[:, :, :1, :]  # sequence x 1 x 3 #Extracting the root axis angles
        root_q_quat = axis_angle_to_quaternion(root_q) #Converting to quaternions
        rotation = torch.Tensor(
            [0.7071068, 0.7071068, 0, 0]
        )  # 90 degrees about the x axis
        root_q_quat = quaternion_multiply(rotation, root_q_quat)
        root_q = quaternion_to_axis_angle(root_q_quat) #Back to quaternions
        local_q[:, :, :1, :] = root_q #Assign new rotated root
       # don't forget to rotate the root position too 😩
        pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
        root_pos = pos_rotation.transform_points(
            root_pos
        )  # basically (y, z) -> (-z, y), expressed as a rotation for readability
    # do FK
    # local_q: axis angle rotations for local rotation of each joint 
    # root_pos: root-joint positions

    positions, rotations = smpl.forward(local_q, root_pos)  # batch x sequence x 24 x 3
    rotations = torch.stack(rotations).permute(1, 2, 0, 3) #Reorder global joint rotations

    return positions, rotations

def smplTo6d(pos, q, scale, aist = True):
    smpl = SMPLSkeleton()
    # to Tensor
    pos /= scale #Normalize by scale
    root_pos = torch.Tensor(np.array([pos]))
    local_q = torch.Tensor(np.array([q]))
    # to ax
    bs, sq, c = local_q.shape
    local_q = local_q.reshape((bs, sq, -1, 3))
    if aist:
        # AISTPP dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
        root_q = local_q[:, :, :1, :]  # sequence x 1 x 3 #Extracting the root axis angles
        root_q_quat = axis_angle_to_quaternion(root_q) #Converting to quaternions
        rotation = torch.Tensor(
            [0.7071068, 0.7071068, 0, 0]
        )  # 90 degrees about the x axis
        root_q_quat = quaternion_multiply(rotation, root_q_quat)
        root_q = quaternion_to_axis_angle(root_q_quat) #Back to quaternions
        local_q[:, :, :1, :] = root_q #Assign new rotated root
       # don't forget to rotate the root position too 😩
        pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
        root_pos = pos_rotation.transform_points(
            root_pos
        )  # basically (y, z) -> (-z, y), expressed as a rotation for readability
    # do FK

    local_q = local_q[:, :, 0:len(smpl_offsets), :]

    # local_q: axis angle rotations for local rotation of each joint 
    # root_os: root-joint positions
    
    root_pos = root_pos[0]
    local_q = local_q[0].reshape(-1, local_q[0].shape[1]*local_q[0].shape[2])

    l = torch.cat((root_pos, local_q), dim=1)

    return l

def create_middle_marker(positions, indices):
    r"""
    Create a virtual marker between two other markers. 
        - positions: object holding marker positions (N_samples, n_markers, 3)
        - indices: an array of integers indicating two markers
    """
    markers = positions[:,indices,:]
    mid_point = np.mean([markers[:, 0, :], markers[:, 1, :]], axis=0)
    mid_point = torch.tensor(mid_point)
    return mid_point

def differentiate_fast(d, order, sr):
    cutoff = .2;
    b, a = butter(2, cutoff, btype='lowpass', analog=False, output='ba')  # Butterworth filter coefficients
    for _ in range(order):
        d = np.diff(d, axis=0) #Difference between consecutive frames
        d = np.concatenate((np.tile(np.array([d[0]]), (1, 1)), d), axis=0) #Repeat first frame
        d = filtfilt(b, a, d, axis=0, padtype=None, padlen=0) #Butterworth filter
    d = d * (sr ** order)
    return d

def translate(df, offsets):
    df[:,0:df.shape[1]:3] = df[:,0:df.shape[1]:3] + offsets[0];
    df[:,1:df.shape[1]:3] = df[:,1:df.shape[1]:3] + offsets[1];
    df[:,2:df.shape[1]:3] = df[:,2:df.shape[1]:3] + offsets[2];
    return(df)

def center_mean(df):
    x = torch.mean(torch.mean(df[:,0:df.shape[1]:3], dim = 0))
    y = torch.mean(torch.mean(df[:,1:df.shape[1]:3], dim = 0))
    z = torch.mean(torch.mean(df[:,2:df.shape[1]:3], dim = 0))
    print("got here")
    df = translate(df, [-x, -y, -z]);
    return(df)

def visu(file, sr):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation
    import pandas as pd
    import numpy as np
    
    try:
        positions = pd.read_pickle(file).to_numpy()
#        positions = pd.read_csv(file)
        #positions = positions.to_numpy()
        positions, _ = smplToPosition(positions[:,0:3], positions[:,3:75], 1, aist = False)
        positions = positions[0]

        #positions = positions.to_numpy()
        #positions = positions.reshape(-1, positions.shape[1]*positions.shape[2])
    except:
        positions = np.load(file, allow_pickle=True)
    
    N = positions.shape[0]  # number of timesteps
#    M = int(positions.shape[1] / 3)  # number of markers
    M = int(positions.shape[1])  # number of markers

    # Reshape positions to have dimensions (N, M, 3)
#    positions = positions.reshape(N, M, 3)

    # Create a figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initialize empty lines and markers for each marker
    lines = [ax.plot([], [], [])[0] for _ in range(M)]
    markers = [ax.plot([], [], [], marker='o', color='blue')[0] for _ in range(M)]
    
    # Highlight the last two markers in red
    markers[-1].set_markerfacecolor('red')
    markers[-2].set_markerfacecolor('red')

    markers[7].set_markerfacecolor('red')
    markers[8].set_markerfacecolor('red')
    markers[10].set_markerfacecolor('red')
    markers[11].set_markerfacecolor('red')

    # Initialize empty lines for the bones (connections between parent-child joints)
    bones = [ax.plot([], [], [], color='blue')[0] for _ in range(M) if smpl_parents[_] != -1]

    # Set axis limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # Animation function to update the plot
    def update(frame):
        for i in range(M):
            # Update marker positions
            x = positions[frame, i, 0]
            y = positions[frame, i, 1]
            z = positions[frame, i, 2]
            lines[i].set_data([x], [y])
            lines[i].set_3d_properties([z])
            markers[i].set_data([x], [y])
            markers[i].set_3d_properties([z])
            
            # Draw lines between each marker and its parent
            if smpl_parents[i] != -1:
                parent_idx = smpl_parents[i]
                parent_x = positions[frame, parent_idx, 0]
                parent_y = positions[frame, parent_idx, 1]
                parent_z = positions[frame, parent_idx, 2]
                bones[i-1].set_data([x, parent_x], [y, parent_y])
                bones[i-1].set_3d_properties([z, parent_z])
        
        return lines + markers + bones

    # Create animation
    fr = sr
    interval = 1000 / fr
    ani = FuncAnimation(fig, update, frames=N, blit=True, interval=interval)

    plt.show()

#all_vids = os.listdir("/Users/pdealcan/Documents/github/EDGEk/data/processed/train/output/")
#index = np.random.randint(0, 100)
#print(all_vids[index])
#visu(f"/Users/pdealcan/Documents/github/EDGEk/data/processed/train/output/{all_vids[index]}", 30)
#visu(f"./current_pred.csv", 30)

visu(f"./data/test/motions_sliced/sliced_1_CLIO_Outsai_poses.pkl", 30)
#visu(f"./data/train/motions_sliced/sliced_0_gBR_sBM_cAll_d04_mBR1_ch04.pkl", 30)


