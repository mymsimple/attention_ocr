import cv2,numpy as np

# 图像缩放，高度都是32,这次的宽度，会和这个批次最宽的图像对齐填充padding
def read_and_resize_image(image_names: list,conf):

    resized_images = []
    padded_images = []

    max_w = 0
    for image_name in image_names:
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        h,w,_ = image.shape
        ratio = conf.INPUT_IMAGE_HEIGHT/h
        image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        h,w,_ = image.shape
        if w > max_w: max_w = w
        resized_images.append(image)

    for resized_image in resized_images:
        dim_difference = conf.INPUT_IMAGE_WIDTH - resized_image.shape[1]

        if (dim_difference<0):
            # 如果图像宽了，就直接resize到最大
            padded_image = cv2.resize(resized_image,(conf.INPUT_IMAGE_WIDTH,conf.INPUT_IMAGE_HEIGHT))
        else:
            # 否则，就给填充0
            padded_image = np.pad(resized_image, [(0, 0),(0, dim_difference),(0,0)], 'constant')

        padded_images.append(padded_image)

    return np.stack(padded_images,axis=0)
