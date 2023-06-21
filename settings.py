from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
# WEBCAM = 'Webcam'
# RTSP = 'RTSP'
YOUTUBE = 'YouTube'

SOURCES_LIST = [IMAGE, VIDEO, YOUTUBE]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'basic.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'detect.jpg'
DEFAULT_SEG_IMAGE = IMAGES_DIR / 'seg.jpg'
DEFAULT_DESEG_IMAGE = IMAGES_DIR / 'detection_seg.jpg'
SMART_DEFAULT_IMAGE = IMAGES_DIR / 'trash.jpg'
SMART_DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'trash-det.jpg'
SMART_DEFAULT_SEG_IMAGE = IMAGES_DIR / 'water.jpg'
SMART_DEFAULT_DESEG_IMAGE = IMAGES_DIR / 'water-seg.jpg'

# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEO_1_PATH = VIDEO_DIR / 'pool.mp4'
VIDEO_2_PATH = VIDEO_DIR / 'pool2.mp4'
VIDEO_3_PATH = VIDEO_DIR / 'pool3.mp4'
VIDEO_4_PATH = VIDEO_DIR / 'pool4.mp4'
VIDEO_5_PATH = VIDEO_DIR / 'smart1.mp4'
VIDEO_6_PATH = VIDEO_DIR / 'smart2.mp4'
VIDEO_7_PATH = VIDEO_DIR / 'smart3.mp4'
VIDEO_8_PATH = VIDEO_DIR / 'smart4.mp4'

VIDEOS_DICT = {
    'LifeJacket_1': VIDEO_1_PATH,
    'LifeJacket_2': VIDEO_2_PATH,
    'LifeJacket_3': VIDEO_3_PATH,
    'LifeJacket_4': VIDEO_4_PATH,
}
VIDEOS_DICT_2 = {
    'SmartInside_1': VIDEO_5_PATH,
    'SmartInside_2': VIDEO_6_PATH,
    'SmartInside_3': VIDEO_7_PATH,
    'SmartInside_4': VIDEO_8_PATH,    
}
# ML Model config
MODEL_DIR = ROOT / 'weights'



DETECTION_MODEL_X = MODEL_DIR / 'yolov8x.pt'
SEGMENTATION_MODEL_X = MODEL_DIR / 'yolov8x-seg.pt'

DETECTION_MODEL_N = MODEL_DIR / 'yolov8n.pt'
SEGMENTATION_MODEL_N = MODEL_DIR / 'yolov8n-seg.pt'

SMARTDETECTION_MODEL_N = MODEL_DIR / 'smart_n.pt'
SMARTSEGMENTATION_MODEL_N = MODEL_DIR / 'smart_n-seg.pt'

SMARTDETECTION_MODEL_X = MODEL_DIR / 'smart_x.pt'
SMARTSEGMENTATION_MODEL_X = MODEL_DIR / 'smart_x-seg.pt'
# Webcam
WEBCAM_PATH = 0





