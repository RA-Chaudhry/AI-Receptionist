"""
Robust Auto-Restarting Tunnel for AI Receptionist
Starts FastAPI Uvicorn server and maintains a resilient Localtunnel connection.
"""
import os
import re
import subprocess
import sys
import time

def run_uvicorn():
    print("[1/2] Starting FastAPI server on http://127.0.0.1:8000 ...")
    cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
    return subprocess.Popen(cmd)

def run_localtunnel():
    cmd = ["npx", "-y", "localtunnel", "--port", "8000"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

def main():
    print("=" * 60)
    print("🚀 Starting AI Receptionist with Auto-Reconnect Tunnel")
    print("=" * 60)

    uvicorn_proc = run_uvicorn()
    time.sleep(3)

    print("\n[2/2] Launching Localtunnel (with Auto-Reconnect)...")

    try:
        while True:
            lt_proc = run_localtunnel()
            url_found = False
            for line in iter(lt_proc.stdout.readline, ''):
                if not line:
                    break
                print(line, end='', flush=True)
                if "your url is:" in line.lower():
                    url_found = True
                    match = re.search(r'https://[^\s]+', line)
                    if match:
                        tunnel_url = match.group(0)
                        print(f"\n✨ ACTIVE PUBLIC URL: {tunnel_url}")
                        print(f"✨ HEALTH CHECK: {tunnel_url}/health\n")
            
            lt_proc.poll()
            print("\n⚠️  Tunnel disconnected. Reconnecting in 3 seconds...\n")
            time.sleep(3)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down server and tunnel...")
    finally:
        uvicorn_proc.terminate()
        try:
            lt_proc.terminate()
        except Exception:
            pass
        print("Stopped.")

if __name__ == "__main__":
    main()
