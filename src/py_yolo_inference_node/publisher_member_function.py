# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import onnx
import onnx_tensorrt
import rclpy
from rclpy.node import Node
import numpy as np
import cv2

from std_msgs.msg import String
from inference_interfaces.msg import InferenceResult, YoloInferenceBox
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class InferencePubSub(Node):

    def __init__(self, sensor_topic : str, pub_topic : str, model, inference_engine):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(InferenceResult, pub_topic, 10)
        self.sensor_topic = sensor_topic
        self.model = model
        self.inference_engine = inference_engine
        self.cv_bridge = CvBridge()
    
    def sub_callback(self, msg : Image):
        #https://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
        #http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/Image.html
        #https://answers.ros.org/question/64318/how-do-i-convert-an-ros-image-into-a-numpy-array/
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        #?

    
        

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def getArgumentParser() -> argparse.ArgumentParser:
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--sensor_topic","-s",type=str,default="/vimba_front_left_center/image")
    argParser.add_argument("--model_file","-f",type=str,defaultt="model.onnx")
    argParser.add_argument("--inference_device","-d",type=str,default="CUDA:1")
    argParser.add_argument("--pub_topic","-p",type=str,default="/inference_result")

def main(args=None):
    rclpy.init(args=args)
    argParser = getArgumentParser()
    
    inference_parameters = vars(argParser.parse_args(args))
    model = onnx.load(inference_parameters['model_file'])
    inference_engine = onnx_tensorrt.backend.prepare(model,device=inference_parameters['inference_device'])

    inf_pubsub = InferencePubSub(inference_parameters['sensor_topic'],inference_parameters['pub_topic'],model,inference_engine)

    rclpy.spin(inf_pubsub)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    inf_pubsub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
