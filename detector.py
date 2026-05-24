import cv2
import time
import os
from inference_sdk import InferenceHTTPClient


class SharkDetector:
    def __init__(self):
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key="5HEmcFSQOtx7kxJTHQMN",
        )
        self.workspace_name = "romain-geffroy"
        self.workflow_id = "general-segmentation-api"
        self.last_inference_time = 0
        self.inference_interval = 1.0  # Safe network throttling: 1 frame/sec
        self.confidence_threshold = 0.25
        self.target_classes = (
            "shark, fish, stingray, ray, dolphin, sealion, seal, person, swimmer, "
            "surfer, boat, vessel, kayak, paddleboard, baot, pes, prt"
        )
        self.threat_classes = {"shark"}
        self.shark_candidate_classes = {"fish", "stingray", "ray"}
        self.people_or_vessel_classes = {
            "person",
            "swimmer",
            "surfer",
            "boat",
            "vessel",
            "kayak",
            "paddleboard",
            "baot",
            "pes",
            "prt",
        }
        self.cached_predictions = {
            "status": "SAFE",
            "shark_count": 0,
            "surfer_count": 0,
            "boxes": [],
            "detected_classes": [],
        }

    def analyze_frame(self, frame):
        current_time = time.time()
        # Throttling rule check
        if current_time - self.last_inference_time < self.inference_interval:
            return self.cached_predictions

        self.last_inference_time = current_time
        temp_filename = "temp_inference_frame.jpg"

        try:
            # Compress to JPG to limit outbound payload sizes
            cv2.imwrite(temp_filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

            # Run Roboflow Cloud Workflow
            result = self.client.run_workflow(
                workspace_name=self.workspace_name,
                workflow_id=self.workflow_id,
                images={"image": temp_filename},
                parameters={"classes": self.target_classes},
                use_cache=True,
            )

            shark_count = 0
            surfer_count = 0
            detected_boxes = []
            detected_classes = []
            predictions = self._extract_predictions(result)

            if not predictions:
                print(f"[AI CORE WARNING] No predictions parsed from workflow response type: {type(result).__name__}")

            for pred in predictions:
                cls_name = self._normalize_class_name(pred.get("class", pred.get("class_name", pred.get("label", ""))))
                confidence = float(pred.get("confidence", pred.get("score", 0.0)) or 0.0)

                if not cls_name or confidence < self.confidence_threshold:
                    continue

                box = self._extract_box(pred)
                if box is None:
                    continue

                x1, y1, x2, y2 = box
                detected_classes.append(f"{cls_name}:{confidence:.2f}")

                label = cls_name
                if cls_name in self.threat_classes:
                    shark_count += 1
                    color = (0, 0, 255)  # Red danger bounding box
                elif cls_name in self.shark_candidate_classes:
                    shark_count += 1
                    label = "shark candidate"
                    color = (0, 80, 255)  # Orange-red candidate threat box
                elif cls_name in self.people_or_vessel_classes:
                    surfer_count += 1
                    label = "surfer/vessel"
                    color = (255, 255, 0)  # Cyan tracking box
                else:
                    color = (0, 255, 255)  # Yellow secondary indicators

                detected_boxes.append(
                    {
                        "box": [x1, y1, x2, y2],
                        "label": f"{label.upper()} {int(confidence * 100)}%",
                        "color": color,
                    }
                )

            if detected_classes:
                print(f"[AI CORE] Parsed detections: {', '.join(detected_classes)}")

            status = "DANGER" if shark_count > 0 else "SAFE"

            self.cached_predictions = {
                "status": status,
                "shark_count": shark_count,
                "surfer_count": surfer_count,
                "boxes": detected_boxes,
                "detected_classes": detected_classes,
            }

        except Exception as e:
            print(f"[AI CORE ERROR] Serverless execution fault: {e}")
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

        return self.cached_predictions

    def _extract_predictions(self, payload):
        predictions = []

        def walk(node):
            if isinstance(node, dict):
                direct_predictions = node.get("predictions")
                if isinstance(direct_predictions, list):
                    predictions.extend(
                        item for item in direct_predictions if isinstance(item, dict) and self._looks_like_prediction(item)
                    )
                elif isinstance(direct_predictions, dict):
                    walk(direct_predictions)

                if self._looks_like_prediction(node):
                    predictions.append(node)
                    return

                for value in node.values():
                    if isinstance(value, (dict, list)):
                        walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(payload)
        return self._deduplicate_predictions(predictions)

    def _looks_like_prediction(self, item):
        has_class = any(key in item for key in ("class", "class_name", "label"))
        has_confidence = any(key in item for key in ("confidence", "score"))
        has_center_box = all(key in item for key in ("x", "y", "width", "height"))
        has_corner_box = all(key in item for key in ("x1", "y1", "x2", "y2"))
        has_bbox = "bbox" in item or "box" in item
        return has_class and has_confidence and (has_center_box or has_corner_box or has_bbox)

    def _deduplicate_predictions(self, predictions):
        seen = set()
        unique_predictions = []
        for pred in predictions:
            cls_name = self._normalize_class_name(pred.get("class", pred.get("class_name", pred.get("label", ""))))
            confidence = float(pred.get("confidence", pred.get("score", 0.0)) or 0.0)
            box = self._extract_box(pred)
            if box is None:
                continue
            fingerprint = (cls_name, round(confidence, 3), tuple(box))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            unique_predictions.append(pred)
        return unique_predictions

    def _extract_box(self, pred):
        if all(key in pred for key in ("x", "y", "width", "height")):
            x = float(pred.get("x", 0))
            y = float(pred.get("y", 0))
            w = float(pred.get("width", 0))
            h = float(pred.get("height", 0))
            return [int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)]

        if all(key in pred for key in ("x1", "y1", "x2", "y2")):
            return [int(pred["x1"]), int(pred["y1"]), int(pred["x2"]), int(pred["y2"])]

        for key in ("bbox", "box"):
            value = pred.get(key)
            if isinstance(value, dict):
                if all(coord in value for coord in ("x", "y", "width", "height")):
                    x = float(value.get("x", 0))
                    y = float(value.get("y", 0))
                    w = float(value.get("width", 0))
                    h = float(value.get("height", 0))
                    return [int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)]
                if all(coord in value for coord in ("x1", "y1", "x2", "y2")):
                    return [int(value["x1"]), int(value["y1"]), int(value["x2"]), int(value["y2"])]
            elif isinstance(value, (list, tuple)) and len(value) >= 4:
                x1, y1, x2, y2 = value[:4]
                return [int(x1), int(y1), int(x2), int(y2)]

        return None

    def _normalize_class_name(self, class_name):
        normalized = str(class_name or "").strip().lower()
        aliases = {
            "requin": "shark",
            "sharks": "shark",
            "sting ray": "stingray",
            "sea lion": "sealion",
            "sea-lion": "sealion",
            "people": "person",
            "human": "person",
            "swimmers": "swimmer",
            "surfers": "surfer",
            "boats": "boat",
            "vessels": "vessel",
        }
        return aliases.get(normalized, normalized)
