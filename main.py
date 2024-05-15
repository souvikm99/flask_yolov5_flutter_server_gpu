import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import cv2
from aiohttp import web
from av import VideoFrame
import aiohttp_cors
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
import sys
sys.path.append('/home/mtech1/flask_flutter_server/webrtc_server/yolov5')
import torch
import datetime

# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages, LoadStreams
from yolov5.utils.general import LOGGER, check_img_size, non_max_suppression, scale_boxes, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
# from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
# from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision.ops import nms

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

################################################################

model = torch.hub.load('yolov5','custom', path='AppleBananaOrange.pt',force_reload=True,source='local', device='0')

device = select_device('0')
# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names

# initialize deepsort
cfg = get_config()
cfg.merge_from_file('deep_sort_pytorch/configs/deep_sort.yaml')
attempt_download('deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

################################################################



ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform, data_channel=None, unique_ids_per_class=None):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform
        self.data_channel = data_channel
        self.unique_ids_per_class = unique_ids_per_class if unique_ids_per_class is not None else {}

    def detections_to_bbs(self, detections):
        bbs = []
        for index, row in detections.iterrows():
            left = row['xmin']
            top = row['ymin']
            width = row['xmax'] - row['xmin']
            height = row['ymax'] - row['ymin']
            confidence = row['confidence']
            detection_class = row['name']
            bbox = [(left, top, width, height), confidence, detection_class]
            bbs.append(bbox)
        return bbs

    async def recv(self):
        frame = await self.track.recv()

        if self.transform == "cartoon":
            img = frame.to_ndarray(format="bgr24")

            # prepare color
            img_color = cv2.pyrDown(cv2.pyrDown(img))
            for _ in range(6):
                img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
            img_color = cv2.pyrUp(cv2.pyrUp(img_color))

            # prepare edges
            img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_edges = cv2.adaptiveThreshold(
                cv2.medianBlur(img_edges, 7),
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                9,
                2,
            )
            img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

            # combine color and edges
            img = cv2.bitwise_and(img_color, img_edges)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "edges":
            # perform edge detection
            img = frame.to_ndarray(format="bgr24")
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "rotate":
            # rotate image
            img = frame.to_ndarray(format="bgr24")
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
            img = cv2.warpAffine(img, M, (cols, rows))

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "yolo":
            img = frame.to_ndarray(format="bgr24")

            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Define NMS thresholds
            conf_thres = 0.5  # Confidence threshold for predictions
            iou_thres = 0.6   # IoU threshold for NMS
            pred = model(img_rgb)


            # # pred.render()  # Apply rendering to img in-place
            # detections = pred.pandas().xyxy[0]

            if pred is not None and len(pred):

                # Send static data as soon as YOLO is called
                static_data = {"message": "YOLO detection process started", "timestamp": str(datetime.datetime.now())}
                if self.data_channel and self.data_channel.readyState == "open":
                    self.data_channel.send(json.dumps(static_data))
                    print("Static data sent to client.")

                detections2 = pred.xywh[0]
                xywhs = detections2[:, 0:4]
                confs = detections2[:, 4]
                clss = detections2[:, 5]

                # Define a confidence threshold
                conf_threshold = 0.5

                # Filter out predictions with low confidence scores
                high_conf_mask = confs > conf_threshold
                filtered_boxes = xywhs[high_conf_mask]
                filtered_scores = confs[high_conf_mask]
                filtered_class_ids = clss[high_conf_mask]

                # Apply Non-Maximum Suppression (NMS)
                nms_indices = nms(filtered_boxes, filtered_scores, iou_threshold=0.5)
                # print(f'LOL :{ detections2}')
                # pred2 = non_max_suppression(pred.xywh, conf_thres=conf_thres, iou_thres=iou_thres)

                # Use the indices returned by NMS to select the final predictions
                nms_boxes = filtered_boxes[nms_indices]
                nms_scores = filtered_scores[nms_indices]
                nms_class_ids = filtered_class_ids[nms_indices]
                msg = {
                            "bbox": nms_boxes
                            }
                self.send_data(json.dumps(msg))
                                


                # Pass only high confidence detections to deepsort
                outputs = deepsort.update(nms_boxes.cpu(), nms_scores.cpu(), nms_class_ids.cpu(), img)
                print(outputs)
                if len(outputs) > 0:
                    for j, output in enumerate(outputs):
                        # Ensure 'j' is a valid index for nms_scores and nms_class_ids before using it
                        if j >= len(nms_scores) or j >= len(nms_class_ids):
                            print(f"Skipping index {j} as it's out of bounds for the scores or class IDs array.")
                            continue
                        bboxes = output[0:4]
                        id = int(output[4])  # Convert from NumPy int64 to Python int
                        cls = int(output[5])  # Convert from NumPy int64 to Python int
                        conf = nms_scores[j]  # This ensures we are using the NMS filtered scores

                        if conf > conf_threshold:  # Double-check if needed, otherwise redundant
                            c = int(cls)  # integer class
                            label = f'{id} {names[c]} {conf:.2f}'
                            print(label)


                        # Track unique IDs for each class
                        if names[c] not in self.unique_ids_per_class:
                            self.unique_ids_per_class[names[c]] = set()
                        self.unique_ids_per_class[names[c]].add(id)

                        # if save_txt:  # Only write to txt file if detected class is 'person'
                        #     # Ensure numeric values are handled correctly
                        #     bbox_left, bbox_top = output[0], output[1]
                        #     bbox_w, bbox_h = output[2] - output[0], output[3] - output[1]

                        #     # Format and write the output line
                        #     output_line = f"{frame_idx + 1} {id} {bbox_left} {bbox_top} {bbox_w} {bbox_h} -1 -1 -1 -1 {label2}\n"
                        #     with open(txt_path, 'a') as f:
                        #         f.write(output_line)
            else:
                deepsort.increment_ages()
            
            class_counts = {class_name: len(ids) for class_name, ids in self.unique_ids_per_class.items()}
            print("Class counts:", class_counts)
                # Construct the new filename for the counts
            counts_file_path = os.path.join('./', 'Neem'+ '_counts.txt')  

                # Write the counts to the new file
            with open(counts_file_path, 'w') as f:
                for class_name, count in class_counts.items():
                    f.write(f"{class_name}: {count}\n")

            print(f"Counts saved to {counts_file_path}")
            # print(f'DETETCTIONS 2 : {detections2}'

            # bbs = self.detections_to_bbs(detections)

            # # print(bbs)

            # # Initialize a dictionary to keep track of unique IDs per class

            # tracks = tracker.update_tracks(bbs, frame=img_rgb) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
            # for track in tracks:
            #     if not track.is_confirmed():
            #         continue
            #     # print(vars(track))
            #     track_id = track.track_id
            #     class_name = track.det_class
            #     print(f'track id {track_id} : {class_name}')
            #     ltrb = track.to_ltrb()
            #     xmin, ymin, xmax, ymax = int(ltrb[0]), int(
            #         ltrb[1]), int(ltrb[2]), int(ltrb[3])
            #     if class_name not in track_id_per_class:
            #         track_id_per_class[class_name] = set()
            #     track_id_per_class[class_name].add(track_id)

            # # Prepare a dictionary to count unique track IDs per class
            # unique_class_counts = {cls: len(ids) for cls, ids in track_id_per_class.items()}
            # unique_class_counts_str = ', '.join(f"{cls}: {count}" for cls, count in unique_class_counts.items())
            # print("Unique detected objects count per class:", unique_class_counts_str)

            # return frame
        else:
            return frame
        
    def send_data(self, data):
        if self.data_channel and self.data_channel.readyState == "open":
            self.data_channel.send(data)


# async def send_data(self, data):
#     # Check if the data channel is open
#     if self.track and self.track.readyState == "open":
#         # Send the data
#         await self.track.send(data)
#     else:
#         print("Data channel is not open")

async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    logger.info(f"{pc_id} Created")

    data_channel = pc.createDataChannel("data")


    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    recorder = MediaBlackhole()

    @data_channel.on("open")
    def on_open():
        async def send_data():
            for i in range(100):  # Example: sending numbers 0 to 99
                if data_channel.readyState == "open":
                    data_channel.send(str(i))
                    await asyncio.sleep(1)  # Sends data every second
        asyncio.ensure_future(send_data())
        

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)


    @pc.on("datachannel")
    def on_datachannel(channel):
        log_info("Data channel opened")
        channel.on("message", on_message)

    def on_message(message):
        log_info(f"Received message {message}")
        
    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)
        # Initialize unique_ids_per_class for each new connection
        unique_ids_per_class = {}

        if track.kind == "audio":
            pc.addTrack(player.audio)  # Relay audio back to the client
            recorder.addTrack(track)
        elif track.kind == "video":
            # Create a video processing track but do not send it back
            video_track = VideoTransformTrack(relay.subscribe(track), transform=params["video_transform"], data_channel=data_channel, unique_ids_per_class=unique_ids_per_class)

            async def process_video():
                try:
                    while True:
                        frame = await video_track.recv()  # Processed frame
                        # Convert BGR to RGB
                        # img = frame.to_ndarray(format="bgr24")

                        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # pred = model(img_rgb)
                        # pred.render()  # Apply rendering to img in-place
                        # detections = pred.pandas().xyxy[0]

                        # bbs = self.detections_to_bbs(detections)
                        # No frame is sent back. Processing results are handled here.
                        
                except Exception as e:
                    log_info("Video processing stopped: %s", str(e))

            # Start the video processing task in the background
            asyncio.create_task(process_video())

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


app = web.Application()
cors = aiohttp_cors.setup(app)
app.on_shutdown.append(on_shutdown)
app.router.add_get("/", index)
app.router.add_get("/client.js", javascript)
app.router.add_post("/offer", offer)

for route in list(app.router.routes()):
    cors.add(route, {
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=5050, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument(
        "--debug", type=bool, default=True, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )


