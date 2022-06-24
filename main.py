import argparse
import os

import cv2
import dlib
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101

from networks.models import build_model
from pred import pred


class Args:
    encoder = 'resnet50_GN_WS'
    decoder = 'fba_decoder'
    weights = './models/FBA.pth'


def apply_deeplab(deeplab, img, device):
    """
    Apply deeplab to an image and return the segmentation mask.
    :param deeplab: deeplab model
    :param img: input image
    :param device: device
    :return: mask
    """
    input_tensor = deeplab_preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = deeplab(input_batch.to(device))['out'][0]
    output_predictions = output.argmax(0).cpu().numpy()
    return output_predictions == 15


def make_deeplab(torch_device):
    """
    Build deeplab model.
    :param torch_device: device
    :return: deeplab model
    """
    dl = deeplabv3_resnet101(pretrained=True).to(torch_device)
    dl.eval()
    return dl


def get_faces(src_img):
    """
    Get faces from an image.
    :param src_img: input image
    :return: faces
    """
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    face = detector(img)
    return face


deeplab_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dlib.cuda.set_device(0)
detector = dlib.get_frontal_face_detector()
# Setup cuda
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
deeplab = make_deeplab(device)
args = Args()
model = build_model(args).to("cuda" if torch.cuda.is_available() else "cpu")


def gen_image(f_name, out_name, bg_name):
    """
    Generate an image of the foreground subject and an image of the background from an input image.
    :param f_name: input image
    :param out_name: output image (without extension)
    :param bg_name: background image (without extension)
    :return: None
    """
    img_orig = cv2.imread(f_name, 1)

    if img_orig.shape[0] > 1200:
        width = int(img_orig.shape[1] * 1200 / img_orig.shape[0])
        img_orig = cv2.resize(img_orig, (width, 1200))
    if img_orig.shape[1] > 1200:
        height = int(img_orig.shape[0] * 1200 / img_orig.shape[1])
        img_orig = cv2.resize(img_orig, (1200, height))

    face = get_faces(img_orig)

    if len(face) != 1:
        return

    # setup image for mask
    k = min(1.0, 1024 / max(img_orig.shape[0], img_orig.shape[1]))
    img = cv2.resize(img_orig, None, fx=k, fy=k, interpolation=cv2.INTER_LANCZOS4)
    mask = apply_deeplab(deeplab, img, device)
    trimap = np.zeros((mask.shape[0], mask.shape[1], 2))
    trimap[:, :, 1] = mask > 0
    trimap[:, :, 0] = mask == 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    trimap[:, :, 0] = cv2.erode(trimap[:, :, 0], kernel)
    trimap[:, :, 1] = cv2.erode(trimap[:, :, 1], kernel)

    # Split image
    fg, bg, alpha = pred((img / 255.0)[:, :, ::-1], trimap, model)

    img_ = img_orig.astype(np.float32) / 255
    alpha_ = cv2.resize(alpha, (img_.shape[1], img_.shape[0]), cv2.INTER_LANCZOS4)
    fg_alpha = np.concatenate([img_, alpha_[:, :, np.newaxis]], axis=2)
    fg_img = (fg_alpha * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    os.makedirs(os.path.dirname(bg_name), exist_ok=True)
    print(out_name + ".png")
    cv2.imwrite(out_name + ".png", fg_img)
    cv2.imwrite(bg_name + "_bg.png", (bg * 255).astype(np.uint8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--infile', default='./test.jpg')
    parser.add_argument('--outfile', default='./test_out.png')

    args = parser.parse_args()
    out, ext = os.path.splitext(args.outfile)
    gen_image(args.infile, out, out)
    