from gpiozero import MotionSensor
from picamera import PiCamera
import socket
import time
import numpy as np
import asyncio
import cv2 as cv
import websockets
import ssl

print(cv.__version__)

client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
context = ssl.create_default_context()
client.connect(("kocjancic.ddns.net", 3001))
sslSocket = context.wrap_socket(client, server_hostname="kocjancic.ddns.net")
print(sslSocket.version())


ipSocket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
ipSocket.connect(("8.8.8.8",80))
thisIP = ipSocket.getsockname()[0]
ipSocket.close()
print(thisIP)

#connection = client.makefile('wb')
connection = sslSocket.makefile('wb')

face_cascade_name = "/usr/local/lib/python3.7/dist-packages/cv2/data/haarcascade_frontalface_alt.xml"
face_cascade = cv.CascadeClassifier()

if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)

pir = MotionSensor(23)

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 20
camera.start_preview()
time.sleep(2)
imageArray = np.empty((480, 640, 3), dtype=np.uint8)


async def recordAndCapture():
    websocket : websockets.WebSocketClientProtocol = await websockets.connect("wss://kocjancic.ddns.net:3000")
    await websocket.send(f'ip/{thisIP}')
    print("sent ip")
    while True:
        print("came here")
        pir.wait_for_motion()
        print("motion detected")
        camera.start_recording(connection,format='h264')
        while pir.motion_detected:
            await record_session(websocket)
        print("motion stopped")
        camera.stop_recording()


async def record_session(websocket):
    camera.wait_recording(2)
    camera.capture(imageArray, 'bgr')
    print("camera captured")
    faceFrames = detectAndDisplay(imageArray)
    print(f"there are {len(faceFrames)} faces")
    for i in range(len(faceFrames)):
        print("sending a faceFrame")
        convertedImage = cv.resize(faceFrames[i],(256,256))
        convertedImage = cv.cvtColor(convertedImage,cv.COLOR_BGR2RGB)
        await websocket.send(convertedImage.tobytes())


def detectAndDisplay(frame):
    faceimg = []
    faces = face_cascade.detectMultiScale(frame)
    for (x, y, w, h) in faces:
        faceimg.append(frame[y:y+h, x:x+w])
    return faceimg


loop = asyncio.get_event_loop()
loop.run_until_complete(recordAndCapture())
