"""Run a shell command on a pod through RunPod's ssh.runpod.io proxy, which forces an interactive PTY.
Usage: python render/proxy_run.py <user@host> '<command>' [timeout_s]"""
import subprocess, sys, time, select, os
target, cmd = sys.argv[1], sys.argv[2]
timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 120
p = subprocess.Popen(["ssh", "-tt", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=25", "-o", "BatchMode=yes",
                      "-i", os.path.expanduser("~/.ssh/id_ed25519"), target],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
buf = b""; t0 = time.time(); sent = False
while time.time() - t0 < timeout:
    r, _, _ = select.select([p.stdout], [], [], 0.5)
    if r:
        chunk = os.read(p.stdout.fileno(), 65536)
        if not chunk: break
        buf += chunk
    if not sent and buf.rstrip().endswith(b"#"):
        p.stdin.write((cmd + "; echo __RC=$?__; exit\n").encode()); p.stdin.flush(); sent = True
    if sent and b"__RC=" in buf and p.poll() is not None: break
try: p.terminate()
except Exception: pass
out = buf.decode(errors="replace").replace("\r", "")
start = out.find(cmd[:40]); print(out[start + len(cmd):] if start >= 0 else out[-2000:])
