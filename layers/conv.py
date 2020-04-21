from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.backend import squeeze
import logging

logger = logging.getLogger(__name__)




'''
    使用resnet50的卷积，因此也使用他的inputs，
    [32,256]的图片，经过后，会变成[1,8,2048]的图像
    输出的结果=>[8,2048]
'''
class Conv():

    def squeeze_wrapper(self, tensor):
        return squeeze(tensor, axis=1)

    def build(self):
        resnet50_model = ResNet50(include_top=False,weights='imagenet',input_shape=(32,256,3))
        x = resnet50_model.output
        conv_outputs = Lambda(self.squeeze_wrapper)(x)

        inputs = resnet50_model.inputs

        # inputs=[(-1,32,256,3)]
        # conv_outputs=[-1,8,2048]
        return conv_outputs,inputs[0]