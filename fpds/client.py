import cv2
import aiohttp
import asyncio
import concurrent.futures
import argparse
import numpy as np


async def camera_source(queue: asyncio.Queue, threadpool: concurrent.futures.ThreadPoolExecutor, src_id: int=0):
    loop = asyncio.get_running_loop()
    try:
        src = await loop.run_in_executor(threadpool, lambda: cv2.VideoCapture(src_id))
        while True:
            _, im = await loop.run_in_executor(threadpool, src.read)
            _, im = await loop.run_in_executor(threadpool, lambda: cv2.imencode('.jpg', im))
            await queue.put(im.tobytes())
    except asyncio.CancelledError:
        pass
    finally:
        src.release()

async def preview_window(queue: asyncio.Queue, threadpool: concurrent.futures.ThreadPoolExecutor):
    loop = asyncio.get_running_loop()
    try:
        while True:
            im = await queue.get()
            im = np.frombuffer(im, dtype=np.uint8)
            im = await loop.run_in_executor(threadpool, lambda: cv2.imdecode(im, cv2.IMREAD_ANYCOLOR))
            cv2.imshow('fpds_remote_preview', im)
            cv2.waitKey(1)
    except asyncio.CancelledError:
        pass
    finally:
        cv2.destroyAllWindows()

async def run_client(
        ws: aiohttp.ClientWebSocketResponse,
        threadpool: concurrent.futures.ThreadPoolExecutor
    ) -> None:
    # --
    loop = asyncio.get_running_loop()
    src_queue = asyncio.Queue(maxsize=1)
    dst_queue = asyncio.Queue(maxsize=1)
    src_task = asyncio.create_task(camera_source(src_queue, threadpool))
    dst_task = asyncio.create_task(preview_window(dst_queue, threadpool))
    try:
        while True:
            im = await src_queue.get()
            await ws.send_bytes(im)
            im = await ws.receive_bytes(timeout=1.0)
            await dst_queue.put(im)
    except asyncio.CancelledError:
        await ws.send_str('close')
        src_task.cancel()
        dst_task.cancel()
        await asyncio.wait([src_task, dst_task])

async def amain(url: str):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as threadpool:
        async with aiohttp.ClientSession() as session, session.ws_connect(url) as ws:
            await run_client(ws, threadpool)

def main():
    parser = argparse.ArgumentParser('fpds.client')
    parser.add_argument('url', type=str, help='WebSocket endpoint of fpds.server e.g. http://localhost:8080/fpds')
    args = parser.parse_args()
    loop = asyncio.get_event_loop()
    task = loop.create_task(amain(args.url))
    try:
        loop.run_until_complete(task)
    except KeyboardInterrupt:
        task.cancel()
        loop.run_until_complete(asyncio.wait_for(task, timeout=None))
    finally:
        loop.close()

if __name__ == '__main__':
    main()
