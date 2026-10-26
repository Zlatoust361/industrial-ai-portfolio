"""vision_bridge: bridges an image stream + text query into open-vocabulary detections.

Executor choice
----------------
`main()` below spins this node on `rclpy.executors.SingleThreadedExecutor`,
not `MultiThreadedExecutor` -- a deliberate choice, not the default left
unexamined:

- The Jetson Orin Nano has 6 ARM Cortex-A78AE cores shared with the rest of
  the robot's ROS2 graph (other perception/control nodes, plus DDS
  discovery traffic, which is itself CPU-sensitive once several nodes are
  running on a small core count). `MultiThreadedExecutor` spins up a pool
  of OS threads for callback dispatch; on a 6-core SoC already hosting
  multiple nodes, that thread/scheduling overhead is a cost this node
  doesn't need to pay.
- The node's only expensive callback is `_on_image` (one `detect()` call
  per frame), and it is inherently serial: only one detection can run on
  the shared GPU/inference engine at a time, so a second callback thread
  waiting on the same accelerator would block rather than add real
  parallelism. Multi-threading a compute-bound Python callback also fights
  the GIL for no benefit.
- `_on_text_query` is cheap (updates one string attribute) and infrequent
  (a human or planner changing what to look for, not a per-frame stream),
  so letting it queue briefly behind an in-flight image callback under a
  single-threaded executor is a non-issue -- there is no responsiveness
  requirement here that multi-threading would actually buy.

If a future backend needs genuine concurrent work (e.g. overlapping
network I/O), reach for a `ReentrantCallbackGroup` scoped to just that
callback plus a `MultiThreadedExecutor` -- not a blanket switch for the
whole node.
"""

from __future__ import annotations

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_bridge_msgs.msg import BoundingBox as BoundingBoxMsg
from vision_bridge_msgs.msg import DetectionArray, DetectionResult

from vision_bridge.detection import BoundingBox, Detection, Detector
from vision_bridge.detector_factory import (
    DEFAULT_MODEL_NAME,
    DEFAULT_SCORE_THRESHOLD,
    build_detector,
)
from vision_bridge.image_codec import image_message_to_array


def bounding_box_to_msg(bbox: BoundingBox) -> BoundingBoxMsg:
    """Convert a plain `Detection.bbox` into the generated `BoundingBox` message."""
    message = BoundingBoxMsg()
    message.x_min = bbox.x_min
    message.y_min = bbox.y_min
    message.x_max = bbox.x_max
    message.y_max = bbox.y_max
    return message


def detection_to_msg(detection: Detection) -> DetectionResult:
    """Convert a plain `Detection` into the generated `DetectionResult` message."""
    message = DetectionResult()
    message.label = detection.label
    message.score = detection.score
    message.bbox = bounding_box_to_msg(detection.bbox)
    return message


class VisionBridgeNode(Node):
    """Subscribes to an image topic + text query, publishes `DetectionArray`.

    Parameters:
        image_topic (str): Topic to subscribe to for `sensor_msgs/Image`
            frames. Default `/camera/image_raw`.
        text_query (str): Initial open-vocabulary text query. Default `"object"`.
        text_query_topic (str): Topic (relative to this node) accepting
            `std_msgs/String` to update the active query at runtime, so the
            query can come from a launch-time parameter, `ros2 param set`,
            *or* another node publishing on this topic. Default `~/text_query`.
        detections_topic (str): Topic (relative to this node) this node
            publishes `vision_bridge_msgs/DetectionArray` on. Default
            `~/detections`.
        detector_backend (str): One of `detector_factory.BACKENDS`
            (`auto`/`tensorrt`/`pytorch`/`stub`). Default `auto`: try
            `engine_path` if set, fall back to PyTorch OWL-ViT otherwise.
        engine_path (str): Path to a TensorRT `.engine` built on *this*
            device (see `ros2_ws/scripts/convert_owlvit_tensorrt.py` --
            engines don't transfer across GPUs/TensorRT versions). Default
            `""` (no engine configured).
        owlvit_model_name (str): HuggingFace Hub id of the OWL-ViT
            checkpoint. Default `google/owlvit-base-patch32`.
        score_threshold (float): Minimum detection confidence to report.
            Default `0.1`.

    See the module docstring for why this node is spun on a
    `SingleThreadedExecutor`.
    """

    def __init__(self, detector: Detector | None = None) -> None:
        """Args:
            detector: The detection backend to use; if `None` (the normal
                case), one is built from the `detector_backend`/`engine_path`/
                `owlvit_model_name`/`score_threshold` parameters via
                `detector_factory.build_detector`. Passing one explicitly is
                mainly for tests/dev (e.g. injecting a `StubDetector`
                without touching parameters).
        """
        super().__init__("vision_bridge")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("text_query", "object")
        self.declare_parameter("text_query_topic", "~/text_query")
        self.declare_parameter("detections_topic", "~/detections")
        self.declare_parameter("detector_backend", "auto")
        self.declare_parameter("engine_path", "")
        self.declare_parameter("owlvit_model_name", DEFAULT_MODEL_NAME)
        self.declare_parameter("score_threshold", DEFAULT_SCORE_THRESHOLD)

        image_topic = self.get_parameter("image_topic").value
        text_query_topic = self.get_parameter("text_query_topic").value
        detections_topic = self.get_parameter("detections_topic").value

        if detector is not None:
            self._detector: Detector = detector
        else:
            self._detector = build_detector(
                backend=self.get_parameter("detector_backend").value,
                engine_path=self.get_parameter("engine_path").value or None,
                model_name=self.get_parameter("owlvit_model_name").value,
                score_threshold=self.get_parameter("score_threshold").value,
                log=self.get_logger().info,
            )
        self._current_query: str = self.get_parameter("text_query").value

        self._image_sub = self.create_subscription(Image, image_topic, self._on_image, 10)
        self._query_sub = self.create_subscription(
            String, text_query_topic, self._on_text_query, 10
        )
        self._detections_pub = self.create_publisher(DetectionArray, detections_topic, 10)

        self.get_logger().info(
            f"vision_bridge up: image_topic={image_topic!r}, "
            f"text_query_topic={text_query_topic!r}, initial query={self._current_query!r}"
        )

    def _on_text_query(self, msg: String) -> None:
        """Update the active text query from `text_query_topic`."""
        if msg.data != self._current_query:
            self.get_logger().info(
                f"Text query updated: {self._current_query!r} -> {msg.data!r}"
            )
        self._current_query = msg.data

    def _on_image(self, msg: Image) -> None:
        """Run detection on an incoming frame and publish the results."""
        try:
            image = image_message_to_array(msg)
        except ValueError as exc:
            self.get_logger().warning(str(exc))
            return

        detections = self._detector.detect(image, self._current_query)

        array_msg = DetectionArray()
        array_msg.header = msg.header
        array_msg.query = self._current_query
        array_msg.detections = [detection_to_msg(d) for d in detections]
        self._detections_pub.publish(array_msg)


def main(args: list[str] | None = None) -> None:
    """Entry point: spin `VisionBridgeNode` on a `SingleThreadedExecutor`."""
    rclpy.init(args=args)
    node = VisionBridgeNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        # rclpy's own SIGINT handling can already call rcl_shutdown() on this
        # context before this KeyboardInterrupt unwinds here (confirmed on a
        # real `ros2 run` Ctrl+C: `rclpy.shutdown()` raised `RCLError: failed
        # to shutdown: rcl_shutdown already called on the given context`) --
        # `rclpy.ok()` reports whether that already happened.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
