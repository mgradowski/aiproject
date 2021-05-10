import argparse
import asyncio
import concurrent.futures
import logging
import pathlib
import sys
import torch
import aiohttp.web
import aiohttp
import numpy as np
import cv2
from typing import Awaitable, Callable


def process_test(im: bytes) -> bytes:
    print(f'TEST :: processing image of size {sys.getsizeof(im)}')
    im = np.frombuffer(im, dtype=np.uint8)
    im = cv2.imdecode(im, cv2.IMREAD_ANYCOLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
    _, im = cv2.imencode('.jpg', im, enc_param)
    return im.tobytes()

def mk_process_yolo(yolofile):
    model = torch.load(yolofile)['model'].fuse().autoshape()

    def process_func(im: bytes) -> bytes:
        im = np.frombuffer(im, dtype=np.uint8)
        im = cv2.imdecode(im, cv2.IMREAD_ANYCOLOR)
        result = model(im)
        result.print()
        im, = result.render()
        enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
        _, im = cv2.imencode('.jpg', im, enc_param)
        return im.tobytes()
    
    return process_func


def mk_image_handler(threadpool: concurrent.futures.ThreadPoolExecutor, process_func: Callable) -> Awaitable:
    async def fpds_handler(request: aiohttp.web.Request) -> aiohttp.web.StreamResponse:
        loop = asyncio.get_running_loop()
        ws = aiohttp.web.WebSocketResponse()
        await ws.prepare(request)

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT and msg.data == 'close':
                if msg.data == 'close':
                    await ws.close()
            elif msg.type == aiohttp.WSMsgType.BINARY:
                response = await loop.run_in_executor(threadpool, lambda: process_func(msg.data))
                await ws.send_bytes(response)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logging.info(f'WebSocket connection closed with exception {ws.exception()}')

        logging.info('WebSocket connection closed')
        return ws
    return fpds_handler

def main() -> None:
    parser = argparse.ArgumentParser('fpds.server')
    parser.add_argument('model_path', type=pathlib.Path, help='Path to a yolov5 model file e.g. aiproject/weights/last060.pt')
    args = parser.parse_args()

    with open(args.model_path, mode='rb') as yolofile:
        process_yolo = mk_process_yolo(yolofile)

    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as threadpool:
        app = aiohttp.web.Application()
        app.add_routes([
            aiohttp.web.get('/test', mk_image_handler(threadpool, process_test)),
            aiohttp.web.get('/yolo', mk_image_handler(threadpool, process_yolo)),
        ])
        aiohttp.web.run_app(app)

if __name__ == '__main__':
    main()