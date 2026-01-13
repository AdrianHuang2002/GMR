import pathlib

HERE = pathlib.Path(__file__).parent
IK_CONFIG_ROOT = HERE / "ik_configs"
ASSET_ROOT = HERE / ".." / "assets"

ROBOT_XML_DICT = {
    "unitree_g1": ASSET_ROOT / "unitree_g1" / "g1_mocap_29dof.xml",
}

IK_CONFIG_DICT = {
    "smplx":{
        "unitree_g1": IK_CONFIG_ROOT / "smplx_to_g1.json",
    },
    "optitrack":{
        "unitree_g1": IK_CONFIG_ROOT / "optitrack_to_g1.json",
    },
}

ROBOT_BASE_DICT = {
    "unitree_g1": "pelvis",
}

VIEWER_CAM_DISTANCE_DICT = {
    "unitree_g1": 2.0,
}