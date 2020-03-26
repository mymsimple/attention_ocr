import cv2,numpy as np
import logging

logger = logging.getLogger(__name__)

# 图像缩放，高度都是32,这次的宽度，会和这个批次最宽的图像对齐填充padding
def read_and_resize_image(image_names: list,conf):

    padded_images = []

    for image_name in image_names:
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        h,w,_ = image.shape
        ratio = conf.INPUT_IMAGE_HEIGHT/h # INPUT_IMAGE_HEIGHT 默认为32
        image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)

        # resize后，看实际宽度和要求的宽度，默认是256
        dim_difference = conf.INPUT_IMAGE_WIDTH - image.shape[1]
        if (dim_difference<0):
            # 如果图像宽了，就直接resize到最大
            padded_image = cv2.resize(image,(conf.INPUT_IMAGE_WIDTH,conf.INPUT_IMAGE_HEIGHT))
        else:
            # 否则，就给填充白色,[(0, 0),(0, dim_difference),(0,0)]=>[高前后忽略,宽前忽略尾部加，通道前后忽略]
            padded_image = np.pad(image, [(0, 0),(0, dim_difference),(0,0)], 'constant',constant_values=(255))

        # cv2.imwrite("data/test.jpg", padded_image)
        padded_images.append(padded_image)

    images = np.stack(padded_images,axis=0)
    # logger.debug("图像的shape：%r",images.shape)
    return images


if __name__=="__main__":

    from main import conf
    read_and_resize_image("data/test.jpg",conf)