import numpy as np

# Static global variables    dictionary
_globals = {
    "CATEGORIES_MERGE_MAP_": {
        # coarse map
        # 0 unlabelled
        # 1 all the ground
        # 2 Three-dimensional obstacle
        # 3 Predestrian
        # 4 Transportation tools, include rider
        # 11 sky
        # 22 other, but carla grassland is wrong labeled as 22, so we use 1 to replace 22
        0: 1,
        22: 1,
        2: 1,
        10: 1,
        23: 1,
        24: 1,
        25: 1,
        27: 1,
        3: 2,
        4: 2,
        5: 2,
        6: 2,
        7: 2,
        8: 2,
        9: 2,
        20: 2,
        21: 2,
        26: 2,
        28: 2,
        12: 3,
        13: 4,
        14: 4,
        15: 4,
        16: 4,
        17: 4,
        18: 4,
        19: 4,
    },
    "CATEGORIES_MERGE_MAP": {
        # 0
        # 1
        # 2
        # 3
        # 4
        0: 1,
        22: 1,
        2: 1,
        10: 1,
        23: 1,
        24: 1,
        25: 1,
        27: 1,
        3: 2,
        26: 2,
        4: 3,
        5: 3,
        28: 3,
        6: 4,
        7: 5,
        8: 5,
        9: 6,
        12: 7,
        13: 8,
        14: 8,
        15: 8,
        16: 8,
        17: 8,
        18: 8,
        19: 8,
    },
    "LABEL_COLORS": np.array(
        [
            (255, 255, 255),  # Unlabeled
            (221, 221, 233),  # Roads
            (244, 35, 232),  # SideWalks
            (70, 70, 70),  # Building
            (102, 102, 156),  # Wall
            (190, 153, 153),  # Fence
            (153, 153, 153),  # Pole
            (250, 170, 30),  # TrafficLight
            (220, 220, 0),  # TrafficSign
            (107, 142, 35),  # Vegetation
            (152, 251, 152),  # Terrain
            (70, 130, 180),  # Sky
            (220, 20, 60),  # Pedestrian
            (255, 0, 0),  # Rider
            (0, 0, 142),  # Car
            (0, 0, 70),  # Truck
            (0, 60, 100),  # Bus
            (0, 80, 100),  # Train
            (0, 0, 230),  # Motorcycle
            (119, 11, 32),  # Bicycle
            (110, 190, 160),  # Static
            (170, 120, 50),  # Dynamic
            (55, 90, 80),  # Other
            (45, 60, 150),  # Water
            (157, 234, 50),  # RoadLine
            (81, 0, 81),  # Ground
            (150, 100, 100),  # Bridge
            (230, 150, 140),  # RailTrack
            (180, 165, 180),  # GuardRail
        ]
    )
    / 255.0,  # normalize each channel [0-1] since is what Open3D uses,
    "CATEGOTIES": {
        15: "bus",
        14: "car",
        16: "truck",
        18: "motorcycle",
        19: "bicycle",
        13: "Pedestrian",
    },
    "CATEGOTIES_INDEX": {
        "bus": 15,
        "car": 14,
        "truck": 16,
        "motorcycle": 18,
        "bicycle": 19,
        "Pedestrian": 13,
    },
    # transform matrix, STANDARD means NUSCENES
    "CAMERA_STANDARD_TO_CARLA": np.array(
        [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    ),
    "EGO_CARLA_TO_STANDARD": np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    ),
    "LIDAR_CARLA_TO_STANDARD": np.array(
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    ),
    "WORLD_CARLA_TO_STANDARD": np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    ),
}


def get_global(name):
    return _globals[name]


def set_global(name, value):  # 添加全局变量
    assert name not in _globals.keys()
    _globals[name] = value
