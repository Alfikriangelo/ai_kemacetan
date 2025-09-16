// src/types/index.ts

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectedObject {
  classId: number;
  confidence: number;
  boundingBox: BoundingBox;
}

export interface Yolov8Output {
  predictions: number[][];
}
