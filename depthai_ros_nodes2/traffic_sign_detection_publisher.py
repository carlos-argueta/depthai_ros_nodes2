import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from depthai_ros_msgs.msg import SpatialDetection, SpatialDetectionArray
from vision_msgs.msg import ObjectHypothesis, BoundingBox2D

from cv_bridge import CvBridge, CvBridgeError

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time



class TrafficSignDetector(Node):
    def __init__(self):
        super().__init__('traffic_sign_detector')

        # Get the parameters:
        self.cam_id = ''
        if self.has_parameter('~cam_id'):
            self.cam_id = self.get_parameter('~cam_id')

        self.camera_name = self.get_parameter_or("~camera_name", Parameter("~camera_name", Parameter.Type.STRING, "default_cam")).value
        self.camera_height_from_floor = 390
        self.source_frame = self.camera_name+"_right_camera_optical_frame"

        self.bridge = CvBridge()

        self.rgb_image_pub = self.create_publisher(Image, '/'+self.camera_name+'/rgb/image', 5)
        self.dets_pub = self.create_publisher(SpatialDetectionArray, '/'+self.camera_name+'/detections/traffic_sign_detections', 5)

        self.labelMap = ['nothing','crosswalk ahead','give way','green light','priority road','red light','right turn','stop sign','traffic light','yellow light']
            
        self.timer_ = self.create_timer(0.2, self.timer_callback)

        self.debug = False

    def timer_callback(self):
        
        self.get_logger().info('Publishing detections: "%s"' % self.counter)
        

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

        depthFrame = depth.getFrame() # depthFrame values are in millimeters

        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        detections = inDet.detections

        det_boxes = []  

        # If the frame is available, draw bounding boxes on it and show the frame
        height = frame.shape[0]
        width  = frame.shape[1]
        for detection in detections:
            roiData = detection.boundingBoxMapping
            roi = roiData.roi
            roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
            topLeft = roi.topLeft()
            bottomRight = roi.bottomRight()
            xmin = int(topLeft.x)
            ymin = int(topLeft.y)
            xmax = int(bottomRight.x)
            ymax = int(bottomRight.y)
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), self.color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

            # Denormalize bounding box
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)


            # Create and transform points
            camera_point = PointStamped()
            camera_point.header.frame_id = self.source_frame
            camera_point.header.stamp = self.get_clock().now().to_msg()
            camera_point.point.x = detection.spatialCoordinates.x / 1000.0;
            camera_point.point.y = (detection.spatialCoordinates.y + self.camera_height_from_floor) / 1000.0;
            camera_point.point.z = detection.spatialCoordinates.z / 1000.0;

            base_point = camera_point

            # Add detection boxes for later publishing
            det_box = [x1, y1, x2, y2] # The bounding box
            pos_box = [base_point.point.x,base_point.point.y,base_point.point.z ] # The actual 3D position
            det_boxes.append((det_box,pos_box,detection.label,detection.confidence * 100.0))


            try:
                label = self.labelMap[detection.label]
            except:
                label = detection.label
            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            print("Distance", detection.spatialCoordinates.z)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), cv2.FONT_HERSHEY_SIMPLEX)

        if self.debug:
            cv2.putText(frame, "NN fps: {:.2f}".format(self.fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))
            cv2.imshow("depth", depthFrameColor)
            cv2.imshow("preview", frame)


        # Publish messages
        rgb_image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="passthrough")
        rgb_image_msg.header.stamp = self.get_clock().now().to_msg()
        rgb_image_msg.header.frame_id = self.camera_name+"_rgb_camera_optical_frame"

        self.rgb_image_pub.publish(rgb_image_msg)

        if det_boxes:
                

            detections_msg = self.bboxToRosMsg(det_boxes)
            detections_msg.header = rgb_image_msg.header
            
            self.dets_pub.publish(detections_msg)

    def bboxToRosMsg(self, boxesData):
        
        opDetectionMsg = SpatialDetectionArray ()

            
        # setting the header
        opDetectionMsg.header.stamp = self.get_clock().now().to_msg()
        opDetectionMsg.header.frame_id = self.camera_name+"_rgb_camera_optical_frame"

        for i, (bbox, position, class_id, score) in enumerate(boxesData):

            xMin = bbox[0]
            yMin = bbox[1]
            xMax = bbox[2]
            yMax = bbox[3]

            xSize = float(xMax - xMin)
            ySize = float(yMax - yMin)
            xCenter = xMin + xSize / 2.0
            yCenter = yMin + ySize / 2.0

            detection = SpatialDetection()
                
            result = ObjectHypothesis()
            #print(t.label)
            result.id = str(class_id)
            result.score = score
            detection.results.append(result)

            detection.bbox.center.x = xCenter
            detection.bbox.center.y = yCenter
            detection.bbox.size_x = xSize
            detection.bbox.size_y = ySize

            detection.position.x = position[0]
            detection.position.y = position[1]
            detection.position.z = position[2]
           
            detection.is_tracking = False
            detection.tracking_id = "-1"

            opDetectionMsg.detections.append(detection)

        
        return opDetectionMsg  

    def set_queues(self, device):
        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        self.previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        self.depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        self.startTime = time.monotonic()
        self.counter = 0
        self.fps = 0
        self.color = (255, 255, 255)

    def get_pipeline(self):
        self.get_logger().info("Creating pipeline")
        
        # Get argument first
        nnBlobPath = str((Path(__file__).parent / Path('models/205+track1_all_signs_3500_openvino_2021.4_5shave.blob')).resolve().absolute())


        if not Path(nnBlobPath).exists():
            import sys
            raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

        
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

        spatialDetectionNetwork.setBlobPath(nnBlobPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)

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

def main(args=None):
    
    rclpy.init(args=args)
    node = TrafficSignDetector()

    pipeline = node.get_pipeline()

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:

        node.set_queues(device)

        while rclpy.ok():
            
            rclpy.spin_once(node)

            if cv2.waitKey(1) == ord('q'):
                break



            

    rclpy.shutdown()

if __name__ == '__main__':
    main()
