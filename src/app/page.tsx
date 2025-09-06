"use client";

import { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";

export default function Home() {
  const [model, setModel] = useState<tf.GraphModel | null>(null);

  useEffect(() => {
    const loadModel = async () => {
      const m = await tf.loadGraphModel("/tfjs_model/model.json");
      setModel(m);
      console.log("✅ Model loaded", m);
    };
    loadModel();
  }, []);

  return (
    <main className="flex min-h-screen items-center justify-center">
      <h1 className="text-2xl font-bold">
        {model ? "Model Loaded 🚀" : "Loading model..."}
      </h1>
    </main>
  );
}
