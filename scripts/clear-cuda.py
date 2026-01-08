import subprocess
import os

def get_gpu_processes():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], stdout=subprocess.PIPE)
    pids = result.stdout.decode('utf-8').strip().split('\n')
    return [int(pid) for pid in pids if pid]

def kill_processes(pids):
    for pid in pids:
        try:
            os.kill(pid, 9)
            print(f"Killed process {pid}")
        except ProcessLookupError:
            print(f"Process {pid} not found")

def release_cuda_memory():
    pids = get_gpu_processes()
    if pids:
        print("Killing the following processes using the GPU:", pids)
        kill_processes(pids)
    else:
        print("No processes using the GPU found.")
    # Clear the CUDA cache
    import torch
    torch.cuda.empty_cache()
    print("All CUDA memory has been released.")

if __name__ == "__main__":
    release_cuda_memory()
