import time
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="True"

from ultralytics import YOLO

model = YOLO("yolo11l.pt")

# times = []
model.track(source=0, show=True)

# timeStart = time.time()
# results = model(["image.jpeg"])
# timeEnd = time.time()
# times.append(timeEnd - timeStart)
# print(len(results))
# results[0].save(filename="result.jpg")

# for result in results:
#     print(result)
# print(times)
# print()
# print(sum(times)/len(times))