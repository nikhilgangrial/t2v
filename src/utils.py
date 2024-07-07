from torch.nn import functional as F
from constants import FP16
import sys
import os

def update_path():
    current = os.path.dirname(__file__)

    sys.path.append(current + "/show_1")
    sys.path.append(current + "/rethinkvsralignment")
    sys.path.append(current + "/practical_rife")
    sys.path.append(current + "/practical_rife/train_log_RIFE")
    sys.path.append(current + "/practical_rife/train_log_SAFA")

    print("appended to path")

def pad_image(img, padding, sr=False):
    if sr:
        if FP16:
            return F.pad(img, padding, mode="reflect").half()
        else:
            return F.pad(img, padding, mode="reflect")
    else:
        if FP16:
            return F.pad(img, padding).half()
        else:
            return F.pad(img, padding)
