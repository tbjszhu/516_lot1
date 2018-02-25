# this code is used for showing keypoints in the image #
import cv2
import argparse
from utils import SIFT_descriptor_generator, brief_descriptor_generator, orb_descriptor_generator
from matplotlib import pyplot as plt
from scipy.misc import imsave

def main (path, descpt_tpye, save_path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    kp = []
    if descpt_tpye == 'orb':
        kp,_ = orb_descriptor_generator(img, 10000)
    img = cv2.imread(path)
    if descpt_tpye == 'sift':
        kp,_ = SIFT_descriptor_generator(img, 10000)
    img = cv2.imread(path)
    if descpt_tpye == 'brief':
        kp,_ = brief_descriptor_generator(img, 10000)
        
    print ("{} key points are detected.".format(len(kp)))
    img = cv2.drawKeypoints(img, kp, img)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    imsave(save_path,img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=str, default='./algo_unit_test/demo.png',
                        help="image path")
    parser.add_argument("-d", type=str, default='sift',
                        help="descriptor type")
    parser.add_argument("--s", type=str, default='./rapport/draw_output.png',
                        help="save path")
    args = parser.parse_args()

    img_path = args.p
    des_type = args.d
    save = args.s

    main(img_path,des_type,save)