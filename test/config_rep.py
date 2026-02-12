import numpy as np

ROBOT = "g1" # Robot name, "go2", "b2", "b2w", "h1", "go2w", "g1"
# Actuator mode: "affine" = general/affine actuators (g1_body29_xhand_mod_waist_new_arms.xml),
#   ctrl = position targets + gravity comp offset; "motor" = motor actuators,
#   ctrl = PD torques + gravity comp torque (scene includes _motor body).
ACTUATOR_MODE = "affine"  # "affine" | "motor"
ROBOT_SCENE = "../unitree_robots/" + ROBOT + "/scene_g1_29dof_motor.xml"  # used when ACTUATOR_MODE == "motor"
ROBOT_SCENE_AFFINE = "../unitree_robots/" + ROBOT + "/scene_g1_29dof.xml"  # used when ACTUATOR_MODE == "affine"


MAX_EP_LENGTH = 1000
# TRAJ_FILE = "/home/tairanh/Workspace/dexmachina/dexmachina/test/g1_xhand_rollout.npz"
TRAJ_FILE = "./g1_xhand_rollout.npz"

PRINT_SCENE_INFORMATION = True # Print link, joint and sensors information of robot
ENABLE_ELASTIC_BAND = False# True # Virtual spring band, used for lifting h1
FIX_FREE = True
ENABLE_GRAVITY_COMPENSATION = False # True # Enable gravity compensation feedforward
GRAVITY_COMPENSATION_SCALE = 0.9 # Scale factor for gravity compensation (0.0 to 1.0)

# SIMULATE_DT = 0.01  # Need to be larger than the runtime of viewer.sync()
SIMULATE_DT = 1/(30*2)  #0.03556  # Need to be larger than the runtime of viewer.sync()
SUBSTEPS = 2
VIEWER_DT = 0.04333  # 50 fps for viewer
# VIEWER_DT = 0.0166  # 50 fps for viewer

WAIST_GROUP = {
    "joint_exprs": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
    "kp": [300, 300, 300],
    "kd": [3.0, 3.0, 3.0],
    "force_range": [88, 50, 50],
    "damping": [0.0, 0.0, 0.0],
    "armature": [0.0, 0.0, 0.0],
    "frictionloss": [0.0, 0.0, 0.0],
    }

ARM_GROUP = {
    "joint_exprs": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"],
    "kp": [80, 80, 80, 80, 40, 40, 40, 80, 80, 80, 80, 40, 40, 40],
    "kd": [3.0, 3.0, 3.0, 3.0, 1.5, 1.5, 1.5, 3.0, 3.0, 3.0, 3.0, 1.5, 1.5, 1.5],
    "force_range": [25, 25, 25, 25, 25, 5, 5, 25, 25, 25, 25, 25, 5, 5],
    "damping": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # -> genesis original
    # "damping": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
    # "damping": [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
    "armature": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02], # -> genesis original
    # "armature": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], 
    # "frictionloss": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    "frictionloss": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # -> genesis original
    # "frictionloss": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}

FINGER_GROUP = {
    "joint_exprs": ["left_hand_thumb_bend_joint", "left_hand_index_bend_joint", "left_hand_mid_joint1", "left_hand_ring_joint1", "left_hand_pinky_joint1", "left_hand_thumb_rota_joint1", "left_hand_index_joint1", "left_hand_mid_joint2", "left_hand_ring_joint2", "left_hand_pinky_joint2", "left_hand_thumb_rota_joint2", "left_hand_index_joint2", "right_hand_thumb_bend_joint", "right_hand_index_bend_joint", "right_hand_mid_joint1", "right_hand_ring_joint1", "right_hand_pinky_joint1", "right_hand_thumb_rota_joint1", "right_hand_index_joint1", "right_hand_mid_joint2", "right_hand_ring_joint2", "right_hand_pinky_joint2", "right_hand_thumb_rota_joint2", "right_hand_index_joint2"],
    "kp": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
    "kd": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    "force_range": [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
    "armature": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], # -> genesis original
    # "armature": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "damping": [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
    "frictionloss": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], # -> genesis original
}

GENESIS_DEFAULTS = {
    "kp": 100.0,
    "kv": 10.0,
    "force_range": 100.0,
    "damping": 1.0, # this is 0.0 in our urdf.
    "armature": 0.1,
    "frictionloss": 0.0,
}


ARM_ACTION_SCALE = 0.05
ARM_ACTION_LOWER_LIMIT = [-3.0892, -1.5882, -2.618, -1.0472, -1.97222, -1.61443, -1.61443, -3.0892, -1.5882, -2.618, -1.0472, -1.97222, -1.61443, -1.61443, -0.70]
ARM_ACTION_UPPER_LIMIT = [2.6704,   2.2515,  2.618,  2.0944,  1.97222,  1.61443,  1.61443,  2.6704,   2.2515,  2.618,  2.0944,  1.97222,  1.61443,  1.61443, 0.70]
HAND_ACTION_SCALE = 0.02
HAND_ACTION_LOWER_LIMIT = [ 0.6,  -1.05, -0.17,    -0.155, 0.1,  0.1,    0.1,  0.1,     0.1,  0.1,     0.1,  0.1, 0.6,  -1.05, -0.17,    -0.155, 0.1,  0.1,    0.1,  0.1,     0.1,  0.1,    0.1,  0.1 ]
HAND_ACTION_UPPER_LIMIT = [1.73,  1.57,  1.83,     0.155, 1.82, 1.82,   1.82, 1.82,    1.82, 1.82,    1.82, 1.82, 1.73,  1.57,  1.83,     0.155, 1.82, 1.82,   1.82, 1.82,    1.82, 1.82,    1.82, 1.82]

DOF_DIM = 53 # full body dof 29 G1 body + 24 hand
ACTION_DIM = 39 # arms + waist_yaw + hands
OBS_CLIP = 5.0

CH_G1_ACTION_INDICES = np.array([11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28, 2])
CH_HAND_ACTION_INDICES = np.array([29, 39, 49, 30, 40, 50, 31, 41, 32, 42, 33, 43, 34, 44, 51, 35, 45, 52, 36, 46, 37, 47, 38, 48])
UNCONTROLLED_INDICES = np.array([i for i in range(53) if i not in CH_G1_ACTION_INDICES and i not in CH_HAND_ACTION_INDICES])
CH_JOINT_NAMES = np.array([
    'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint',  # 0-2
    'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint',   # 3-5
    'left_hip_yaw_joint', 'right_hip_yaw_joint',   'waist_pitch_joint',  # 6-8
    
    'left_knee_joint', 'right_knee_joint', # 9-10
    
    'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', # 11-12
    'left_ankle_pitch_joint', 'right_ankle_pitch_joint', # 13-14
    'left_shoulder_roll_joint', 'right_shoulder_roll_joint', # 15-16
    'left_ankle_roll_joint', 'right_ankle_roll_joint', # 17-18
    'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_elbow_joint', 'right_elbow_joint', # 19-22
    
    'left_wrist_roll_joint', 'right_wrist_roll_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint', # 23-26
    'left_wrist_yaw_joint', 'right_wrist_yaw_joint', # 27-28
    
    'left_hand_thumb_bend_joint', 'left_hand_index_bend_joint', 'left_hand_mid_joint1', # 29-31
        'left_hand_ring_joint1', 'left_hand_pinky_joint1', # 32-33
    'right_hand_thumb_bend_joint', 'right_hand_index_bend_joint', 'right_hand_mid_joint1', # 34-36
        'right_hand_ring_joint1', 'right_hand_pinky_joint1', # 37-38
    
    'left_hand_thumb_rota_joint1', 'left_hand_index_joint1', 'left_hand_mid_joint2', # 39-41
        'left_hand_ring_joint2', 'left_hand_pinky_joint2', # 42-43
    'right_hand_thumb_rota_joint1', 'right_hand_index_joint1', 'right_hand_mid_joint2', # 44-46
        'right_hand_ring_joint2', 'right_hand_pinky_joint2', # 47-48
    
    'left_hand_thumb_rota_joint2', 'left_hand_index_joint2', # 49-50
    'right_hand_thumb_rota_joint2', 'right_hand_index_joint2' # 51-52
])

HISTORY_STEPS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
DEMO_HORIZONS = [1, 2, 4, 8, 16]
OBS_SCALE = {
    'dof_pos_history': 0.1,
    'dof_vel_history': 0.1,
    'prev_actions_history': 0.1,
    'phase_history': 0.1,
    'diff_dof_pos_future': 0.1,

    'part_pos': 1,
    'part_quat': 1,
    'obj_dof_pos': 1,

    'demo_state_diff': 0.5,
    'ref_future_positions': 0.1,
    'diff_curr_pos': 0.1,
}


# Obs indices - for reference
DOF_POS_IDX = np.arange(477)
DOF_VEL_IDX = np.arange(477, 954)
PREV_ACTIONS_IDX = np.arange(954, 1305)
PHASE_IDX = np.arange(1305, 1314)
DIFF_DOF_POS_FUTURE_IDX = np.arange(1314, 1579)

PART_POS_IDX = np.arange(1579, 1585)
PART_QUAT_IDX = np.arange(1585, 1593)
OBJ_DOF_POS_IDX = np.arange(1593, 1594)
DEMO_STATE_DIFF_IDX = np.arange(1594, 1609)
REF_FUTURE_POSITIONS_IDX = np.arange(1609, 1624)
DIFF_CURR_POS_IDX = np.arange(1624, 1627)


## observations. --> initial obs.
## dt
## 




# root pos: init
#    0.0535, -0.2669,  1.0383

# root quat: init
#    0.6843, -0.4121, -0.5801,  0.1593

# articulation init:
#      1.4036

# parts_pos init:
# tensor([[ 0.0535, -0.2669,  1.0383],
#         [ 0.0535, -0.2669,  1.0383]], device='cuda:0')

# parts_quat init:
# tensor([[ 0.6843, -0.4121, -0.5801,  0.1593,  
#        0.6255,  0.0598, -0.7090, -0.3201]], device='cuda:0')

# (Pdb) obs_dict['diff_curr_pos']
# tensor([[0.0000000000e+00, 0.0000000000e+00, 8.1750156824e-04]],
#        device='cuda:0')


# (Pdb) obs_dict['demo_state_diff'] --> the position errors are a bit diverging ~1e-5   reduce friction in mj.
# tensor([[ 4.1207084432e-03,  1.8146216462e-04,  4.7028064728e-05,
#           4.1649844497e-03,  3.6271760473e-04,  9.4056129456e-05,
#           4.0843510069e-03,  8.2056591054e-04, -6.8724155426e-05,
#           3.8536763750e-03,  1.9962626975e-03, -1.8816590309e-03,
#           3.8663113955e-03,  4.2701419443e-03, -3.8820505142e-03]],
#        device='cuda:0')

# Pdb) obs_dict['ref_future_positions']
# tensor([[5.3245467134e-03, -2.6725029573e-02, 1.0383173078e-01, 5.3021120839e-03,
#          -2.6761678979e-02, 1.0383726656e-01, 5.2996235900e-03, -2.6800174266e-02,
#          1.0381650925e-01, 5.3780982271e-03, -2.6799848303e-02, 1.0377071053e-01,
#          5.5026360787e-03, -2.6833448559e-02, 1.0375212878e-01]],
#        device='cuda:0')



# (Pdb) all_obs_dict['g1_xhand_with_history']['prev_actions'].squeeze(0).reshape(9, 39)[0]
# tensor([ 0.0097,  0.1000, -0.0118,  0.1000,  0.0035,  0.0121,  0.0049,  0.1000,
#         -0.0481, -0.0437,  0.0675,  0.0179,  0.0896, -0.0978, -0.0124,  0.0606,
#         -0.0310, -0.0500,  0.0839, -0.1000,  0.0359, -0.1000,  0.0972,  0.0281,
#          0.0416,  0.1000, -0.0511,  0.1000, -0.0252,  0.0337,  0.0433, -0.1000,
#         -0.0863,  0.1000,  0.0799, -0.1000, -0.0844, -0.1000,  0.0444],
#        device='cuda:0')


# with body mj:
# (Pdb) part_pos
# array([ 0.0535, -0.2669,  1.0383,  0.0535, -0.2669,  1.0383])
# (Pdb) part_quat
# array([ 0.684305  , -0.41210301, -0.58010423,  0.15930116, 
#        0.62543474, 0.05979606, -0.70906575, -0.32012743])

# with geom mj:
# (Pdb) part_pos
# array([ 0.03461615, -0.12036612,  0.99122731, -0.01583917, -0.23288254,
#         1.16600811])
# (Pdb) part_quat
# array([-0.13618301,  0.69876416, -0.14017377, -0.68813818,  0.54880903,
#        -0.62156252, -0.34275568,  0.44157358])