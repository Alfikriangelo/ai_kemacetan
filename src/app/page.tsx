"use client";

import { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import { loadModel, predict, hitungLampuPcus } from "@/utils/traffic";
import Image from "next/image";
import { DetectedObject } from "@/types";

interface ImageResult {
  id: string;
  src: string;
  detectedObjects: DetectedObject[];
  counts: Record<string, Record<string, number>>;
  durations: Record<string, number>;
}

const CLASS_NAMES = ["motorcycle", "car"];
const CLASS_COLORS: Record<string, string> = {
  car: "#00FF00",
  motorcycle: "#FFFF00",
  bus: "#0000FF",
  truck: "#FF0000",
  unknown: "#808080",
};

const DEMO_IMAGES = [
  { name: "north", url: "/demo_images/utara.jpg" },
  { name: "west", url: "/demo_images/barat.jpg" },
  { name: "south", url: "/demo_images/selatan.jpg" },
  { name: "east", url: "/demo_images/timur.jpg" },
];

export default function Home() {
  const [model, setModel] = useState<tf.GraphModel | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [processing, setProcessing] = useState<boolean>(false);
  const [imageResults, setImageResults] = useState<ImageResult[]>([]);
  const [totalDurations, setTotalDurations] = useState<Record<
    string,
    number
  > | null>(null);
  const [totalCounts, setTotalCounts] = useState<Record<
    string,
    Record<string, number>
  > | null>(null);

  const canvasRefs = useRef<Record<string, HTMLCanvasElement | null>>({});
  const imageRefs = useRef<Record<string, HTMLImageElement | null>>({});
  const [imagesLoaded, setImagesLoaded] = useState<Record<string, boolean>>({});

  useEffect(() => {
    async function loadModelOnMount() {
      const loadedModel = await loadModel();
      if (!loadedModel) {
        setIsLoading(false);
        return;
      }
      setModel(loadedModel);
      setIsLoading(false);
    }
    loadModelOnMount();
  }, []);

  useEffect(() => {
    imageResults.forEach((result) => {
      const canvas = canvasRefs.current[result.id];
      const image = imageRefs.current[result.id];

      if (
        imagesLoaded[result.id] &&
        canvas &&
        image &&
        result.detectedObjects
      ) {
        drawBoxes(canvas, image, result.detectedObjects);
      }
    });
  }, [imageResults, imagesLoaded]);

  const drawBoxes = (
    canvas: HTMLCanvasElement,
    image: HTMLImageElement,
    detections: DetectedObject[]
  ) => {
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const modelWidth = 640;
    const modelHeight = 640;
    const displayWidth = image.offsetWidth;
    const displayHeight = image.offsetHeight;

    canvas.width = displayWidth;
    canvas.height = displayHeight;

    const scaleX = displayWidth / modelWidth;
    const scaleY = displayHeight / modelHeight;

    detections.forEach((detection) => {
      const { x, y, width, height } = detection.boundingBox;

      const scaledX = x * scaleX;
      const scaledY = y * scaleY;
      const scaledWidth = width * scaleX;
      const scaledHeight = height * scaleY;

      const className = CLASS_NAMES[detection.classId] || "unknown";
      const confidence = (detection.confidence * 100).toFixed(1);

      const boxColor = CLASS_COLORS[className] || CLASS_COLORS["unknown"];

      ctx.beginPath();
      ctx.rect(scaledX, scaledY, scaledWidth, scaledHeight);
      ctx.lineWidth = 2;
      ctx.strokeStyle = boxColor;
      ctx.fillStyle = boxColor + "33";
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = boxColor;
      ctx.font = "16px Arial";
      ctx.fillText(
        `${className} (${confidence}%)`,
        scaledX,
        scaledY > 10 ? scaledY - 5 : scaledY + 15
      );
    });
  };

  const processImages = async (
    imageSources: (string | File)[],
    loadedModel: tf.GraphModel
  ) => {
    setProcessing(true);
    setTotalDurations(null);
    setTotalCounts(null);
    setImageResults([]);
    setImagesLoaded({});

    const processingPromises = imageSources.map((source) => {
      return new Promise<ImageResult | null>((resolve) => {
        const img = new window.Image();
        const src =
          typeof source === "string" ? source : URL.createObjectURL(source);
        const fileName =
          typeof source === "string"
            ? source.split("/").pop()?.split(".")[0] || "unknown"
            : source.name.split(".")[0];
        const imageId = `${fileName}-${Date.now()}`;

        img.onload = async () => {
          const detectedObjects = await predict(img, loadedModel);

          const jalur = fileName;
          const counts: Record<string, number> = {
            car: 0,
            motorcycle: 0,
            bus: 0,
            truck: 0,
          };
          detectedObjects.forEach((obj) => {
            const className = CLASS_NAMES[obj.classId];
            if (className) {
              counts[className] = (counts[className] || 0) + 1;
            }
          });

          const dummyCounts = { [jalur]: counts };
          const durations = hitungLampuPcus(dummyCounts);

          const result: ImageResult = {
            id: imageId,
            src: src,
            detectedObjects,
            counts: dummyCounts,
            durations,
          };
          resolve(result);
        };
        img.onerror = () => {
          console.error(`Failed to load image: ${src}`);
          resolve(null);
        };
        img.src = src;
      });
    });

    const results = (await Promise.all(processingPromises)).filter(
      Boolean
    ) as ImageResult[];

    const allVehicleCounts: Record<string, Record<string, number>> = {};
    results.forEach((result) => {
      const jalur = Object.keys(result.counts)[0];
      if (!allVehicleCounts[jalur]) {
        allVehicleCounts[jalur] = { car: 0, motorcycle: 0, bus: 0, truck: 0 };
      }
      for (const vehicleType in result.counts[jalur]) {
        allVehicleCounts[jalur][vehicleType] +=
          result.counts[jalur][vehicleType];
      }
    });

    const finalDurations = hitungLampuPcus(allVehicleCounts);
    setTotalDurations(finalDurations);
    setTotalCounts(allVehicleCounts);
    setImageResults(results);
    setProcessing(false);
  };

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0 || !model) {
      return;
    }
    const fileArray = Array.from(files);
    processImages(fileArray, model);
  };

  const handleDemoPredict = () => {
    if (!model) {
      return;
    }
    const demoImageUrls = DEMO_IMAGES.map((img) => img.url);
    processImages(demoImageUrls, model);
  };

  const handleImageLoad = (id: string) => {
    setImagesLoaded((prev) => ({ ...prev, [id]: true }));
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-100">
        <p className="text-lg text-gray-700 font-medium">Loading model...</p>
      </div>
    );
  }

  if (!model) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-100">
        <p className="text-lg text-red-500 font-medium">
          Failed to load model. Please check the console.
        </p>
      </div>
    );
  }

  return (
    <main className="min-h-screen flex flex-col items-center p-4 sm:p-8 md:p-12 lg:p-24 bg-gray-50">
      <div className="text-center mb-8">
        <h1 className="text-3xl sm:text-4xl font-bold text-gray-800">
          Traffic Analysis
        </h1>
        <p className="mt-2 text-md text-gray-600">
          Detect vehicles and calculate optimal green light duration.
        </p>
      </div>

      <div className="w-full max-w-2xl bg-white p-6 rounded-xl shadow-lg mb-8">
        <p className="text-lg font-semibold text-gray-700 mb-4">
          Start Analysis
        </p>
        <div className="flex flex-col sm:flex-row gap-4 items-stretch sm:items-center">
          <label
            htmlFor="file-upload"
            className="flex-1 text-center cursor-pointer bg-blue-600 text-white font-semibold py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors"
          >
            Select Images
          </label>
          <input
            id="file-upload"
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            className="hidden"
            multiple
          />
          <button
            onClick={handleDemoPredict}
            className="flex-1 bg-gray-200 text-gray-700 font-semibold py-3 px-6 rounded-lg hover:bg-gray-300 transition-colors"
          >
            Use Demo Images
          </button>
        </div>
      </div>

      {processing && (
        <p className="text-lg font-medium text-blue-600 mt-4 mb-8">
          Processing images...
        </p>
      )}

      {totalDurations && (
        <div className="w-full max-w-4xl bg-white p-6 rounded-xl shadow-lg mb-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">
            Total Analysis Result
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {Object.keys(totalDurations)
              .sort()
              .map((jalur) => (
                <div
                  key={jalur}
                  className="bg-gray-50 p-4 rounded-lg shadow-sm border border-gray-200 text-center"
                >
                  <h3 className="text-lg font-bold text-gray-800 capitalize">
                    {jalur}
                  </h3>
                  {totalCounts && totalCounts[jalur] && (
                    <>
                      <p className="text-sm text-gray-600 mt-1">
                        Total:{" "}
                        {Object.values(totalCounts[jalur]).reduce(
                          (a, b) => a + b,
                          0
                        )}{" "}
                        vehicles
                      </p>
                      <p className="text-xs text-gray-500 italic">
                        ({totalCounts[jalur].car || 0} cars,{" "}
                        {totalCounts[jalur].motorcycle || 0} motorcycles,{" "}
                      </p>
                    </>
                  )}
                  <p className="mt-3 text-lg font-semibold text-gray-800">
                    Duration:{" "}
                    <span className="text-green-600">
                      {totalDurations[jalur].toFixed(2)}s
                    </span>
                  </p>
                </div>
              ))}
          </div>
        </div>
      )}

      <div className="w-full max-w-6xl">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {imageResults.map((result) => (
            <div
              key={result.id}
              className="relative bg-white p-2 rounded-lg shadow-md overflow-hidden"
            >
              <h4 className="text-sm font-semibold text-gray-700 text-center mb-2">
                Image:{" "}
                <span className="capitalize">
                  {result.id.split("-")[0].replace(/-\d+$/, "")}
                </span>
              </h4>
              <div className="relative w-full h-auto">
                <Image
                  id={`image-${result.id}`}
                  src={result.src}
                  alt={`Analysis of ${result.id}`}
                  width={640}
                  height={480}
                  className="rounded-md w-full h-auto"
                  ref={(el) => {
                    imageRefs.current[result.id] = el;
                  }}
                  onLoad={() => handleImageLoad(result.id)}
                />
                <canvas
                  ref={(el) => {
                    canvasRefs.current[result.id] = el;
                  }}
                  className="absolute top-0 left-0"
                  style={{ width: "100%", height: "100%" }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="w-full max-w-2xl mt-12 bg-white p-6 rounded-xl shadow-lg text-center">
        <h2 className="text-xl font-bold text-gray-800 mb-4">
          Download Demo Images
        </h2>
        <p className="text-gray-600 text-sm mb-4">
          Use these images to test on your own device.
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {DEMO_IMAGES.map((img) => (
            <div key={img.name} className="flex flex-col items-center">
              <a
                href={img.url}
                download={`${img.name}.jpg`}
                className="block w-full"
              >
                <Image
                  src={img.url}
                  alt={img.name}
                  width={150}
                  height={150}
                  className="rounded-md hover:scale-105 transition-transform border-2 border-gray-300 w-full h-auto object-cover"
                />
              </a>
              <a
                href={img.url}
                download={`${img.name}.jpg`}
                className="mt-2 text-sm text-blue-600 font-medium hover:underline capitalize"
              >
                {img.name}.jpg
              </a>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
