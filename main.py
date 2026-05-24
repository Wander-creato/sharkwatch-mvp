import asyncio
import base64
import cv2
import json
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from stream_processor import VideoStreamProcessor
from detector import SharkDetector


app = FastAPI(title="SharkWatch NC - Cloud Link Dashboard")
templates = Jinja2Templates(directory="templates")

# Initialize streaming threads and deep-learning configurations
streamer = VideoStreamProcessor().start()
detector = SharkDetector()


@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[SYSTEM] Mission Control interface attached to WebSocket telemetry gateway.")

    try:
        while True:
            ret, frame = streamer.read()
            if not ret or frame is None:
                await asyncio.sleep(0.03)
                continue

            # Pass image tensor directly into the background throttled analysis logic
            ai_results = detector.analyze_frame(frame)

            # Draw real-time bounding overlays via OpenCV
            for item in ai_results["boxes"]:
                x1, y1, x2, y2 = item["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), item["color"], 3)
                cv2.putText(
                    frame,
                    item["label"],
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    item["color"],
                    2,
                )

            # Scale frame dimension to fit web view standard profiles efficiently
            small_frame = cv2.resize(frame, (854, 480))
            _, buffer = cv2.imencode(".jpg", small_frame)
            base64_frame = base64.b64encode(buffer).decode("utf-8")

            payload = {
                "status": ai_results["status"],
                "shark_count": ai_results["shark_count"],
                "surfer_count": ai_results["surfer_count"],
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "image": f"data:image/jpeg;base64,{base64_frame}",
            }

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.04)  # Output target matches standard Web display rates (~25fps)

    except WebSocketDisconnect:
        print("[SYSTEM] Mission Control interface closed connection.")
    except Exception as e:
        print(f"[SYSTEM CRITICAL] Pipeline disruption: {e}")


@app.on_event("shutdown")
def shutdown_event():
    streamer.stop()
    print("[SYSTEM] Clean release of cloud streaming feeds complete.")
