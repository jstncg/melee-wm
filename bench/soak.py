"""Throughput soak against the one-message-per-step protocol, pipelined like the browser does."""
import asyncio, json, sys, time
sys.path.insert(0, "/workspace/pyextra")   # websockets lives here on the pod, so insert before importing it
import websockets
PERIOD, MAX_INFLIGHT, N = 0.100, 4, 200    # MAX_INFLIGHT matches play.html; 2 throttles to ~3.5 steps/s

def script(step):
    p = step % 60
    if p < 12: return ["P1_joy_x_pos"]
    if p < 18: return ["P1_X"]
    if p < 24: return ["P1_joy_x_pos", "P1_A"]
    if p < 34: return ["P1_joy_x_neg"]
    if p < 40: return ["P1_X", "P1_joy_x_neg"]
    if p < 46: return ["P1_B"]
    return []

async def main():
    async with websockets.connect("ws://localhost:8765", max_size=None) as ws:
        got, ms, t0 = 0, [], time.perf_counter()
        async def rx():
            nonlocal got
            while got < N:
                d = await ws.recv()
                hl = int.from_bytes(d[:4], "big")
                ms.append(json.loads(d[4:4+hl])["ms"]); got += 1
        async def tx():
            for i in range(N):
                while got + MAX_INFLIGHT <= i: await asyncio.sleep(0.005)
                await ws.send(json.dumps({"keys": script(i)}))
                await asyncio.sleep(PERIOD)
        await asyncio.gather(tx(), rx())
        wall = time.perf_counter() - t0
        ms.sort()
        print(f"OK  {got} steps in {wall:.1f}s = {got/wall:.1f} steps/s = {got/wall*2:.1f} video fps")
        print(f"    dream time {got*0.1:.1f}s -> {got*0.1/wall:.2f}x real time")
        print(f"    compute ms: median {ms[len(ms)//2]:.0f}  p95 {ms[int(len(ms)*.95)]:.0f}")

asyncio.run(main())
