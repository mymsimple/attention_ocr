import cv2,numpy as np

# 图像缩放，高度都是32
def resize_batch_image(image_names: list):

    resized_images = []
    padded_images = []


    max_w = 0
    for image_name in image_names:
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        h,w,_ = image.shape
        ratio = 32/h
        image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        h,w,_ = image.shape
        if w > max_w: max_w = w
        resized_images.append(image)

    # print("=========w=========",max_w)
    for resized_image in resized_images:
        # print("resized_images.shape:",resized_image.shape)
        # print("max:", max_w)
        # print("width:",resized_image.shape[1])
        dim_difference = max_w - resized_image.shape[1]
        # print("resized_image.shape:", resized_image.shape)
        # print("dim_difference:",dim_difference)
        padded_image = np.pad(resized_image, [(0, 0),(0, dim_difference),(0,0)], 'constant')
        padded_images.append(padded_image)
        # print("padded_image.shape:",padded_image.shape)

    # return padded_images
    return np.stack(padded_images,axis=0)
