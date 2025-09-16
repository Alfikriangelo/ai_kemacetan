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
  { name: "utara", url: "/demo_images/utara.jpg" },
  { name: "barat", url: "/demo_images/barat.jpg" },
  { name: "selatan", url: "/demo_images/selatan.jpg" },
  { name: "timur", url: "/demo_images/timur.jpg" },
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

      // Hanya menggambar jika gambar sudah dimuat
      if (
        imagesLoaded[result.id] &&
        canvas &&
        image &&
        result.detectedObjects
      ) {
        drawBoxes(canvas, image, result.detectedObjects);
      }
    });
  }, [imageResults, imagesLoaded]); // Tambahkan imagesLoaded sebagai dependency

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
    setImagesLoaded({}); // Reset status loading gambar

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
      console.error("Model not loaded yet.");
      return;
    }
    const fileArray = Array.from(files);
    processImages(fileArray, model);
  };

  const handleDemoPredict = () => {
    if (!model) {
      console.error("Model not loaded yet.");
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
      <div className="flex items-center justify-center min-h-screen">
        Memuat model...
      </div>
    );
  }

  if (!model) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        Gagal memuat model. Periksa konsol.
      </div>
    );
  }

  return (
    <main className="flex min-h-screen flex-col items-center p-24">
      <h1 className="text-4xl font-bold mb-8">Sistem Prediksi Lalu Lintas</h1>
      <div className="my-8 flex gap-4">
        <label
          htmlFor="file-upload"
          className="cursor-pointer bg-blue-500 text-white font-bold py-2 px-4 rounded hover:bg-blue-700 transition-colors"
        >
          Pilih Hingga 10 Gambar Lalu Lintas
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
          className="bg-gray-500 text-white font-bold py-2 px-4 rounded hover:bg-gray-700 transition-colors"
        >
          Gunakan Gambar Demo
        </button>
      </div>

      <div className="mt-4 w-full max-w-xl">
        <h2 className="text-xl font-bold mb-2">Download Gambar Demo</h2>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {DEMO_IMAGES.map((img) => (
            <div key={img.name} className="flex flex-col items-center">
              <a href={img.url} download={`${img.name}.png`}>
                <Image
                  src={img.url}
                  alt={img.name}
                  width={150}
                  height={150}
                  className="rounded-md hover:scale-105 transition-transform"
                />
              </a>
              <a
                href={img.url}
                download={`${img.name}.png`}
                className="mt-2 text-sm text-blue-500 hover:underline"
              >
                {img.name}
              </a>
            </div>
          ))}
        </div>
      </div>

      {processing && <p className="text-xl mt-4">Memproses gambar...</p>}

      {totalDurations && (
        <div className="mt-8 p-6 bg-gray-100 dark:bg-zinc-800 rounded-lg shadow-lg w-full max-w-xl">
          <h2 className="text-2xl font-semibold mb-4">Hasil Analisis Total</h2>
          <div className="grid grid-cols-2 gap-4">
            {Object.keys(totalDurations)
              .sort()
              .map((jalur) => (
                <div
                  key={jalur}
                  className="bg-white dark:bg-zinc-700 p-4 rounded-md"
                >
                  <h3 className="text-xl font-bold mb-2">Jalur {jalur}</h3>
                  {totalCounts && totalCounts[jalur] && (
                    <>
                      <p className="text-sm">
                        Total:{" "}
                        {Object.values(totalCounts[jalur]).reduce(
                          (a, b) => a + b,
                          0
                        )}{" "}
                        kendaraan
                      </p>
                      <p className="text-sm italic">
                        ({totalCounts[jalur].car} mobil,{" "}
                        {totalCounts[jalur].motorcycle} motor,{" "}
                        {totalCounts[jalur].bus} bus, {totalCounts[jalur].truck}{" "}
                        truk)
                      </p>
                    </>
                  )}
                  <p className="mt-2">
                    Durasi Lampu Hijau:{" "}
                    <span className="font-semibold text-green-500">
                      {totalDurations[jalur].toFixed(2)} detik
                    </span>
                  </p>
                </div>
              ))}
          </div>
        </div>
      )}

      <div className="mt-8 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {imageResults.map((result) => (
          <div
            key={result.id}
            className="relative border-2 border-gray-400 p-2 rounded-lg"
          >
            <h4 className="text-md font-bold mb-2">
              Gambar: {result.id.split("-")[0].replace(/-\d+$/, "")}
            </h4>
            <Image
              id={`image-${result.id}`}
              src={result.src}
              alt={`Gambar ${result.id}`}
              width={640}
              height={480}
              className="rounded-md"
              style={{ width: "100%", height: "auto" }}
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
              style={{
                top: "2.5rem",
                left: "0.5rem",
                zIndex: 10,
                position: "absolute",
              }}
            />
          </div>
        ))}
      </div>
    </main>
  );
}
