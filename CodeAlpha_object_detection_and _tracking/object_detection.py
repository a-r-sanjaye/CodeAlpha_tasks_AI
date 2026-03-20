import cv2
import torch
import torchvision
from torchvision import transforms as T
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist

from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

# COCO Class labels
# These are the 91 categories (including background) that the Faster R-CNN model 
# was originally trained to detect on the COCO (Common Objects in Context) dataset.
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'clothing', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.rects = OrderedDict()
        self.labels = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid, rect, label):
        self.objects[self.nextObjectID] = centroid
        self.rects[self.nextObjectID] = rect
        self.labels[self.nextObjectID] = label
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.rects[objectID]
        del self.labels[objectID]
        del self.disappeared[objectID]

    def update(self, rects, labels):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], rects[i], labels[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.rects[objectID] = rects[col]
                self.labels[objectID] = labels[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], rects[col], labels[col])

        return self.objects

# Load pre-trained Faster R-CNN
print("[INFO] Loading Faster R-CNN model...")
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

# Initialize tracker
ct = CentroidTracker()

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

print("[INFO] Starting video stream...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to tensor
    img = T.ToTensor()(frame)
    
    # Run Faster R-CNN detection
    with torch.no_grad():
        prediction = model([img])

    rects = []
    curr_labels = []
    # Process detections
    for i in range(len(prediction[0]['boxes'])):
        score = prediction[0]['scores'][i].item()
        if score > 0.7:  # Confidence threshold
            box = prediction[0]['boxes'][i].detach().cpu().numpy()
            label_idx = prediction[0]['labels'][i].item()
            label = COCO_INSTANCE_CATEGORY_NAMES[label_idx]
            
            (startX, startY, endX, endY) = box.astype("int")
            rects.append((startX, startY, endX, endY))
            curr_labels.append(label)

    # Update tracker
    objects = ct.update(rects, curr_labels)

    # Draw tracking information
    for (objectID, centroid) in objects.items():
        startX, startY, endX, endY = ct.rects[objectID]
        label = ct.labels[objectID]
        
        display_text = "{}: ID {}".format(label, objectID)
        
        # Draw bounding box
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # Draw label + ID
        cv2.putText(frame, display_text, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw centroid
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)


    cv2.imshow("Faster R-CNN Object Detection & Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
