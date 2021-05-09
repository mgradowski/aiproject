import logging
import sys
import asyncio
import concurrent.futures
from typing import Awaitable
import aiohttp.web
import aiohttp
import numpy as np
import cv2


def process_image(im: bytes) -> bytes:
    print(f'processing image of size {sys.getsizeof(im)}')
    im = np.frombuffer(im, dtype=np.uint8)
    im = cv2.imdecode(im, cv2.IMREAD_ANYCOLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    _, im = cv2.imencode('.jpg', im)
    return im.tobytes()

def mk_fpds_handler(threadpool: concurrent.futures.ThreadPoolExecutor) -> Awaitable:
    async def fpds_handler(request: aiohttp.web.Request) -> aiohttp.web.StreamResponse:
        loop = asyncio.get_running_loop()
        ws = aiohttp.web.WebSocketResponse()
        await ws.prepare(request)

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT and msg.data == 'close':
                if msg.data == 'close':
                    await ws.close()
            elif msg.type == aiohttp.WSMsgType.BINARY:
                response = await loop.run_in_executor(threadpool, lambda: process_image(msg.data))
                await ws.send_bytes(response)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logging.info(f'WebSocket connection closed with exception {ws.exception()}')

        logging.info('WebSocket connection closed')
        return ws
    return fpds_handler

def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as threadpool:
        app = aiohttp.web.Application()
        app.add_routes([
            aiohttp.web.get('/fpds', mk_fpds_handler(threadpool)),
        ])
        aiohttp.web.run_app(app)

if __name__ == '__main__':
    main()