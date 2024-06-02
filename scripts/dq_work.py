import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", type=str)
parser.add_argument("-q", "--queue", type=str, default="qAll_1")
args = parser.parse_args()
import simple_disk_queue as sdq

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
queue = sdq.DiskQueue(args.queue, overwrite=False, verbose=True)
print(os.environ["CUDA_VISIBLE_DEVICES"], len(queue))
sdq.DiskQueue.run(args.queue, verbose=True, raise_error=False)
