import cv2
import time
import os
from inference_sdk import InferenceHTTPClient


class SharkDetector:
    def __init__(self):
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key="5HEmcFSQOtx7kxJTHQMN"
        )
        self.workspace_name = "romain-geffroy"
        self.workflow_id = "general-segmentation-api"
        self.last_inference_time = 0
        self.inference_interval = 1.0  # Safe network throttling: 1 frame/sec
        self.cached_predictions = {"status": "SAFE", "shark_count": 0, "surfer_count": 0, "boxes": []}

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
                parameters={
                    "classes": "boat, shark, stingray, person, dolphin, baot, kelp, pes, prt, sealion"
                },
                use_cache=True
            )

            shark_count = 0
            surfer_count = 0
            detected_boxes = []

            # Response validation and deep parsing
            if result and "outputs" in result:
                predictions_block = result["outputs"][0]
                for key in predictions_block.keys():
                    if isinstance(predictions_block[key], dict) and "predictions" in predictions_block[key]:
                        preds = predictions_block[key]["predictions"]
                        for pred in preds:
                            cls_name = pred.get("class", "").lower()
                            confidence = pred.get("confidence", 0.0)

                            if confidence > 0.45:
                                x = int(pred.get("x", 0))
                                y = int(pred.get("y", 0))
                                w = int(pred.get("width", 0))
                                h = int(pred.get("height", 0))

                                x1 = int(x - w / 2)
                                y1 = int(y - h / 2)
                                x2 = int(x + w / 2)
                                y2 = int(y + h / 2)

                                label = cls_name
                                if cls_name in ["shark"]:
                                    shark_count += 1
                                    color = (0, 0, 255)  # Red danger bounding box
                                elif cls_name in ["person", "pes", "boat", "baot"]:
                                    surfer_count += 1
                                    label = "surfer/vessel"
                                    color = (255, 255, 0)  # Cyan tracking box
                                else:
                                    color = (0, 255, 255)  # Yellow secondary indicators

                                detected_boxes.append({
                                    "box": [x1, y1, x2, y2],
                                    "label": f"{label.upper()} {int(confidence * 100)}%",
                                    "color": color
                                })

            status = "DANGER" if shark_count > 0 else "SAFE"

            self.cached_predictions = {
                "status": status,
                "shark_count": shark_count,
                "surfer_count": surfer_count,
                "boxes": detected_boxes
            }

        except Exception as e:
            print(f"[AI CORE ERROR] Serverless execution fault: {e}")
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

        return self.cached_predictions
