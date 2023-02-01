#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

import tf2_ros
#import tf2_geometry_msgs

from depthai_ros_msgs.msg import SpatialDetection, SpatialDetectionArray
from vision_msgs.msg import ObjectHypothesis, BoundingBox2D
from geometry_msgs.msg import Pose, PoseArray, PointStamped
from sensor_msgs.msg import CameraInfo, Image

from pathlib import Path

import time

from cv_bridge import CvBridge, CvBridgeError

frameName = ""


class TrafficSignDetector(Node):

    def __init__(self):

        super().__init__('traffic_sign_detector')

        # Get the parameters:
        self.cam_id = ''
        if self.has_parameter('~cam_id'):
            self.cam_id = self.get_parameter('~cam_id')

        self.camera_name = self.get_parameter_or("~camera_name", Parameter("~camera_name", Parameter.Type.STRING, "default_cam")).value

        self.debug = self.get_parameter_or("~debug", Parameter("~debug", Parameter.Type.BOOL, False)).value

        #self.camera_param_uri = self.get_parameter("~camera_param_uri")

        self.camera_height_from_floor = 390
        
        self.rgb_image_pub = self.create_publisher(Image, '/'+self.camera_name+'/rgb/image', 5)
        
        self.dets_pub = self.create_publisher(SpatialDetectionArray, '/'+self.camera_name+'/detections/traffic_sign_detections', 1)
        
        self.tf_buffer = tf2_ros.Buffer() #tf buffer length
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    

        self.bridge = CvBridge()

        
    
        found, device_info = None, None
        if self.cam_id is not None and self.cam_id != '':
            found, device_info = dai.Device.getDeviceByMxId(cam_id)

            if not found:
                raise RuntimeError("Device not found!")
        else:
            print("No camera ID specified, finding one")
            for device in dai.Device.getAllAvailableDevices():
                print(f"{device.getMxId()} {device.state}")
                self.cam_id = device.getMxId()
            if self.cam_id != '':
                print("Using camera ",self.cam_id)
                found, device_info = dai.Device.getDeviceByMxId(self.cam_id)
            else:
                raise RuntimeError("No device found!")

        self.pipeline = self.create_pipeline()

        # Pipeline defined, now the device is assigned and pipeline is started
        with dai.Device(self.pipeline) as device:

        
            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            self.previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            self.detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            #self.xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
            self.depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

            self.startTime = time.monotonic()
            self.counter = 0
            self.fps = 0

            self.seq = 0
            self.sequenceNum = 0

            self.labelMap = ['nothing','crosswalk ahead','give way','green light','priority road','red light','right turn','stop sign','traffic light','yellow light']
            
            #print("Warming up")
            #time.sleep(5)
            #print("Done")
            timer_period = 1.0  # seconds
            self.pub_timer = self.create_timer(timer_period, self.detections_publisher)
            self.pub_i = 0

    def bboxToRosMsg(self, boxesData):
        
        opDetectionMsg = SpatialDetectionArray ()

            
        # setting the header
        opDetectionMsg.header.seq = self.sequenceNum;
        opDetectionMsg.header.stamp = rospy.Time.now()
        opDetectionMsg.header.frame_id = frameName;

        for i, (bbox, position, class_id, score) in enumerate(boxesData):

            xMin = int(bbox[0])
            yMin = int(bbox[1])
            xMax = int(bbox[2])
            yMax = int(bbox[3])

            xSize = xMax - xMin;
            ySize = yMax - yMin;
            xCenter = xMin + xSize / 2.0;
            yCenter = yMin + ySize / 2.0;

            detection = SpatialDetection()
                
            result = ObjectHypothesis()
            #print(t.label)
            result.id = class_id
            result.score = score
            detection.results.append(result)

            detection.bbox.center.x = xCenter;
            detection.bbox.center.y = yCenter;
            detection.bbox.size_x = xSize;
            detection.bbox.size_y = ySize;

            detection.position.x = position[0]
            detection.position.y = position[1]
            detection.position.z = position[2]
           
            detection.is_tracking = False;
            detection.tracking_id = "-1"

            opDetectionMsg.detections.append(detection)

        
        self.sequenceNum += 1

        return opDetectionMsg

    def create_pipeline(self):
        
        nnBlobPath = str((Path(__file__).parent / Path("models/205+track1_all_signs_3500_openvino_2021.4_5shave.blob")).resolve().absolute())
        # MobilenetSSD label texts
        
        syncNN = True

        print("Using models/frozen_inference_graph_openvino_2021.4_5s")

        syncNN = True

        # Create pipeline
        pipeline = dai.Pipeline()

        # Define sources and outputs
        camRgb = pipeline.create(dai.node.ColorCamera)
        spatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)


        xoutRgb = pipeline.create(dai.node.XLinkOut)
        xoutNN = pipeline.create(dai.node.XLinkOut)
        xoutDepth = pipeline.create(dai.node.XLinkOut)

        xoutRgb.setStreamName("rgb")
        xoutNN.setStreamName("detections")
        xoutDepth.setStreamName("depth")

        # Properties
        camRgb.setPreviewSize(300, 300)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)


        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # Setting node configs
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Align depth map to the perspective of RGB camera, on which inference is done
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

        # Setting node configs
        spatialDetectionNetwork.setBlobPath(nnBlobPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)
        spatialDetectionNetwork.setNumInferenceThreads(1)

        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        camRgb.preview.link(spatialDetectionNetwork.input)
        if syncNN:
            spatialDetectionNetwork.passthrough.link(xoutRgb.input)
        else:
            camRgb.preview.link(xoutRgb.input)

        spatialDetectionNetwork.out.link(xoutNN.input)

        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)


        return pipeline

    def detections_publisher(self):

        inPreview = self.previewQueue.get()
        inDet = self.detectionNNQueue.get()
        depth = self.depthQueue.get()

        self.counter+=1
        current_time = time.monotonic()
        if (current_time - self.startTime) > 1 :
            self.fps = self.counter / (current_time - self.startTime)
            self.counter = 0
            self.startTime = current_time

        frame = inPreview.getCvFrame()
        depthFrame = depth.getFrame()

        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        detections = inDet.detections
        '''
        det_boxes = []  
        source_frame = camera_name+"_right_camera_optical_frame"
        rospy_time_now = rospy.Time.now();

        if len(detections) != 0:
            boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
            roiDatas = boundingBoxMapping.getConfigData()

            for roiData in roiDatas:
                roi = roiData.roi
                roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                topLeft = roi.topLeft()
                bottomRight = roi.bottomRight()
                xmin = int(topLeft.x)
                ymin = int(topLeft.y)
                xmax = int(bottomRight.x)
                ymax = int(bottomRight.y)

                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), 255, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

        # If the frame is available, draw bounding boxes on it and show the frame
        height = frame.shape[0]
        width  = frame.shape[1]
        for detection in detections:
            # Denormalize bounding box
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)

            camera_point = PointStamped()
            camera_point.header.frame_id = source_frame;
            camera_point.header.stamp = rospy_time_now;
            camera_point.point.x = detection.spatialCoordinates.x / 1000.0;
            camera_point.point.y = (detection.spatialCoordinates.y + camera_height_from_floor) / 1000.0;
            camera_point.point.z = detection.spatialCoordinates.z / 1000.0;


            # Convert point from camera optical frame to camera frame
            target_frame = camera_name+"_right_camera_frame"
            source_frame = camera_name+"_right_camera_optical_frame"

            transform1 = tf_buffer.lookup_transform(target_frame,
                source_frame, #source frame
                rospy.Time(0), #get the tf at first available time
                rospy.Duration(1.0)) #wait for 1 second
            #frame_point = tf2_geometry_msgs.do_transform_point(camera_point, transform1)

            # Convert the point from camera frame to target frame
            target_frame = "base_link"
            source_frame = camera_name+"_right_camera_frame"
            transform2 = tf_buffer.lookup_transform(target_frame,
                source_frame, #source frame
                rospy.Time(0), #get the tf at first available time
                rospy.Duration(1.0)) #wait for 1 second
            #base_point = tf2_geometry_msgs.do_transform_point(frame_point, transform2)
            base_point = camera_point
           
            det_box = [x1, y1, x2, y2]
            pos_box = [base_point.point.x,base_point.point.y,base_point.point.z ]
            det_boxes.append((det_box,pos_box,detection.label,int(detection.confidence * 100)))

            try:
                label = self.labelMap[detection.label]
            except:
                label = detection.label
            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), cv2.FONT_HERSHEY_SIMPLEX)

        # Create and publish ROS messages
        #depth_frame = (depthFrameColor * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)

        # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
        #cv_frame = cv2.applyColorMap(cv_frame, cv2.COLORMAP_JET)

        #depth_image_msg = bridge.cv2_to_imgmsg(depthFrameColor, encoding="passthrough")
        #depth_image_msg.header.stamp = rospy.Time.now()
        #depth_image_msg.header.seq = seq
        #depth_image_msg.header.frame_id = camera_name+"_rgb_camera_optical_frame"
        
        #depth_image_pub.publish(depth_image_msg)

        rgb_image_msg = bridge.cv2_to_imgmsg(frame, encoding="passthrough")
        rgb_image_msg.header.stamp = rospy.Time.now()
        rgb_image_msg.header.seq = self.seq
        rgb_image_msg.header.frame_id = camera_name+"_rgb_camera_optical_frame"

        self.rgb_image_pub.publish(rgb_image_msg)
            
        if det_boxes:
            

            detections_msg = bboxToRosMsg(det_boxes)
            detections_msg.header = rgb_image_msg.header
            
            self.dets_pub.publish(detections_msg)

            self.seq = self.seq + 1


        if self.debug:
            cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))
            cv2.imshow("depth", depthFrameColor)
            cv2.imshow("preview", frame)
            
        
        

        #if cv2.waitKey(1) == ord('q'):
        #    break

        '''
        
def main(args=None):
  
    rclpy.init(args=args)

    traffic_sign_detector = TrafficSignDetector()

    rclpy.spin(traffic_sign_detector)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    traffic_sign_detector.destroy_node()
    rclpy.shutdown()
    print("bye")
   
if __name__ == '__main__':
    main()
    