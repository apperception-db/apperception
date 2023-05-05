import socket
import subprocess


def report_progress(progress: int, total: int, message: str = ""):
    host = socket.gethostname()

    subprocess.run(
        [
            "ssh",
            "-o StrictHostKeyChecking=no",
            "-o UserKnownHostsFile=/dev/null",
            "chanwutk@freddie.millennium.berkeley.edu",
            "-t",
            f"touch ~/gcp-dashboard/progress__{host} && echo {progress} {total} {message} > ~/gcp-dashboard/progress__{host}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    subprocess.run(
        [
            "ssh",
            "-o StrictHostKeyChecking=no",
            "-o UserKnownHostsFile=/dev/null",
            "chanwutk@freddie.millennium.berkeley.edu",
            "-t",
            f"cd ~/gcp-dashboard && python3 update.py",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
