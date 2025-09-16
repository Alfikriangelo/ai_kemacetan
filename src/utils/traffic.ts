// src/utils/traffic.ts

import * as tf from "@tensorflow/tfjs";
import { BoundingBox, DetectedObject } from "@/types";

const PCU_WEIGHTS = {
  car: 1.0,
  motorcycle: 0.25,
  bus: 2.5,
  truck: 3.0,
};

function hitungVolumePcus(dataJalur: Record<string, number>): number {
  return Object.keys(dataJalur).reduce((total, vehicleType) => {
    const count = dataJalur[vehicleType];
    const weight = PCU_WEIGHTS[vehicleType as keyof typeof PCU_WEIGHTS] || 1.0;
    return total + count * weight;
  }, 0);
}

export function hitungLampuPcus(
  jalurData: Record<string, Record<string, number>>
): Record<string, number> {
  const volumes: Record<string, number> = {};
  for (const jalur in jalurData) {
    volumes[jalur] = hitungVolumePcus(jalurData[jalur]);
  }

  const totalVolume = Object.values(volumes).reduce((sum, v) => sum + v, 0);

  const hasil: Record<string, number> = {};
  for (const jalur in volumes) {
    const v = volumes[jalur];
    if (totalVolume > 0) {
      const waktu = (v / totalVolume) * 100;
      hasil[jalur] = parseFloat(waktu.toFixed(2));
    } else {
      hasil[jalur] = 0;
    }
  }
  return hasil;
}

let yolov8Model: tf.GraphModel | null = null;
const MODEL_PATH = "/tfjs_model/model.json";

export async function loadModel(): Promise<tf.GraphModel | null> {
  if (yolov8Model) {
    return yolov8Model;
  }
  try {
    console.log("Memuat model dari:", MODEL_PATH);
    const model = await tf.loadGraphModel(MODEL_PATH);
    yolov8Model = model;
    console.log("Model berhasil dimuat!");
    return model;
  } catch (error) {
    console.error("Gagal memuat model:", error);
    return null;
  }
}

export async function predict(
  image: HTMLImageElement,
  model: tf.GraphModel,
  confidenceThreshold: number = 0.09,
  iouThreshold: number = 0.45
): Promise<DetectedObject[]> {
  const [modelWidth, modelHeight] = model.inputs[0].shape!.slice(1, 3);
  const detectedObjects: DetectedObject[] = [];

  const tensor = tf.browser
    .fromPixels(image)
    .resizeBilinear([modelHeight, modelWidth])
    .div(255.0)
    .expandDims(0);

  const predictions = model.predict(tensor) as tf.Tensor;

  const transposed = tf.tidy(() => predictions.transpose([0, 2, 1]));

  const boxesAndScores = tf.tidy(() => transposed.squeeze());

  const boxes = tf.tidy(() => boxesAndScores.slice([0, 0], [-1, 4]));
  const classScores = tf.tidy(() => boxesAndScores.slice([0, 4]));

  const maxScores = tf.tidy(() => classScores.max(1).as1D());
  const classIds = tf.tidy(() => classScores.argMax(1).as1D());

  const boxesForNMS = tf.tidy(() => {
    const [x_center, y_center, width, height] = tf.split(boxes, 4, 1);
    const y1 = tf.sub(y_center, tf.div(height, 2));
    const x1 = tf.sub(x_center, tf.div(width, 2));
    const y2 = tf.add(y_center, tf.div(height, 2));
    const x2 = tf.add(x_center, tf.div(width, 2));
    return tf.concat([y1, x1, y2, x2], 1);
  });

  const nonMaxSuppressionTensor = await tf.image.nonMaxSuppressionAsync(
    boxesForNMS as tf.Tensor2D,
    maxScores as tf.Tensor1D,
    100,
    iouThreshold,
    confidenceThreshold
  );

  const selectedIndices = nonMaxSuppressionTensor.arraySync() as number[];
  const boxesData = boxes.arraySync() as number[][];
  const maxScoresData = maxScores.arraySync() as number[];
  const classIdsData = classIds.arraySync() as number[];

  selectedIndices.forEach((selectedIndex: number) => {
    const box = boxesData[selectedIndex];
    const score = maxScoresData[selectedIndex];
    const classId = classIdsData[selectedIndex];

    const [x_center, y_center, width, height] = box;

    const x1 = x_center - width / 2;
    const y1 = y_center - height / 2;

    detectedObjects.push({
      classId: classId,
      confidence: score,
      boundingBox: {
        x: x1,
        y: y1,
        width: width,
        height: height,
      },
    });
  });

  tf.dispose([
    tensor,
    predictions,
    transposed,
    boxesAndScores,
    boxes,
    classScores,
    maxScores,
    classIds,
    boxesForNMS,
    nonMaxSuppressionTensor,
  ]);

  return detectedObjects;
}
