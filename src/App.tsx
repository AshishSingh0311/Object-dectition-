import { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocossd from '@tensorflow-models/coco-ssd';
import * as mobilenet from '@tensorflow-models/mobilenet';
import { Camera, AlertTriangle, BarChart3, Clock, Target, Layers, Shield, Zap, BrainCircuit } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts';

interface Detection {
  class: string;
  score: number;
  bbox?: [number, number, number, number];
}

interface TimeSeriesData {
  timestamp: number;
  detections: number;
  avgConfidence: number;
}

interface ClassificationData {
  className: string;
  probability: number;
}

const COLORS = ['#00FFFF', '#FF00FF', '#FFFF00', '#00FF00', '#FF0000', '#0000FF', '#FFA500', '#800080'];

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [objectModel, setObjectModel] = useState<cocossd.ObjectDetection | null>(null);
  const [classificationModel, setClassificationModel] = useState<mobilenet.MobileNet | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [classifications, setClassifications] = useState<ClassificationData[]>([]);
  const [stats, setStats] = useState<{ name: string; count: number }[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isVideoReady, setIsVideoReady] = useState(false);
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesData[]>([]);
  const [confidenceDistribution, setConfidenceDistribution] = useState<{ range: string; count: number }[]>([]);
  const [totalDetections, setTotalDetections] = useState(0);
  const [averageConfidence, setAverageConfidence] = useState(0);
  const [fps, setFps] = useState(0);
  const [processingTime, setProcessingTime] = useState(0);

  useEffect(() => {
    const loadModels = async () => {
      try {
        await tf.ready();
        const [loadedObjectModel, loadedClassificationModel] = await Promise.all([
          cocossd.load({
            base: 'mobilenet_v2'
          }),
          mobilenet.load({
            version: 2,
            alpha: 1.0
          })
        ]);
        setObjectModel(loadedObjectModel);
        setClassificationModel(loadedClassificationModel);
        setIsLoading(false);
      } catch (err) {
        setError('Failed to load the detection models');
        setIsLoading(false);
      }
    };
    loadModels();
  }, []);

  useEffect(() => {
    if (!objectModel || !classificationModel) return;

    const setupCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'environment',
            width: { ideal: 1920 },
            height: { ideal: 1080 }
          },
          audio: false,
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.addEventListener('loadeddata', () => {
            setIsVideoReady(true);
          });
        }
      } catch (err) {
        setError('Unable to access camera');
      }
    };

    setupCamera();
  }, [objectModel, classificationModel]);

  useEffect(() => {
    if (!objectModel || !classificationModel || !videoRef.current || !canvasRef.current || !isVideoReady) return;

    const videoWidth = videoRef.current.videoWidth;
    const videoHeight = videoRef.current.videoHeight;
    canvasRef.current.width = videoWidth;
    canvasRef.current.height = videoHeight;

    let frameCount = 0;
    let lastTime = performance.now();
    let processingStartTime = 0;

    const detectFrame = async () => {
      if (!videoRef.current || !canvasRef.current) return;

      processingStartTime = performance.now();

      // Run object detection and classification in parallel
      const [predictions, classificationResult] = await Promise.all([
        objectModel.detect(videoRef.current),
        classificationModel.classify(videoRef.current)
      ]);

      const currentTime = performance.now();
      frameCount++;

      // Calculate FPS every second
      if (currentTime - lastTime >= 1000) {
        setFps(Math.round((frameCount * 1000) / (currentTime - lastTime)));
        frameCount = 0;
        lastTime = currentTime;
      }

      // Calculate processing time
      setProcessingTime(performance.now() - processingStartTime);

      setDetections(predictions);
      setClassifications(classificationResult);

      // Update statistics
      const counts = predictions.reduce((acc: { [key: string]: number }, pred) => {
        acc[pred.class] = (acc[pred.class] || 0) + 1;
        return acc;
      }, {});

      const newStats = Object.entries(counts).map(([name, count]) => ({
        name,
        count,
      }));
      setStats(newStats);

      // Update time series data
      const currentConfidence = predictions.reduce((acc, pred) => acc + pred.score, 0) / (predictions.length || 1);
      setTimeSeriesData(prev => {
        const newData = [...prev, {
          timestamp: Date.now(),
          detections: predictions.length,
          avgConfidence: currentConfidence
        }];
        return newData.slice(-60); // Keep last 60 data points (1 minute)
      });

      // Update confidence distribution
      const confidenceRanges = predictions.reduce((acc: { [key: string]: number }, pred) => {
        const range = Math.floor(pred.score * 10) * 10;
        const rangeStr = `${range}-${range + 10}%`;
        acc[rangeStr] = (acc[rangeStr] || 0) + 1;
        return acc;
      }, {});

      setConfidenceDistribution(
        Object.entries(confidenceRanges).map(([range, count]) => ({
          range,
          count,
        }))
      );

      // Update total detections and average confidence
      setTotalDetections(prev => prev + predictions.length);
      setAverageConfidence(currentConfidence);

      // Draw detections
      const ctx = canvasRef.current.getContext('2d')!;
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      
      predictions.forEach(prediction => {
        if (!prediction.bbox) return;
        
        const [x, y, width, height] = prediction.bbox;
        const confidence = Math.round(prediction.score * 100);
        
        // Calculate color based on confidence
        const hue = confidence * 1.2; // Higher confidence = more green
        ctx.strokeStyle = `hsla(${hue}, 100%, 50%, 0.8)`;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);

        // Draw background for text
        ctx.fillStyle = `hsla(${hue}, 100%, 50%, 0.7)`;
        const padding = 4;
        const textWidth = ctx.measureText(`${prediction.class} ${confidence}%`).width;
        ctx.fillRect(x - padding, y - 25, textWidth + (padding * 2), 25);
        
        // Draw text
        ctx.fillStyle = '#FFFFFF';
        ctx.font = '16px Arial';
        ctx.fillText(
          `${prediction.class} ${confidence}%`,
          x,
          y - 8
        );
      });

      requestAnimationFrame(detectFrame);
    };

    detectFrame();
  }, [objectModel, classificationModel, isVideoReady]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white text-2xl animate-pulse">Loading AI models...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-red-500 flex items-center gap-2">
          <AlertTriangle />
          <span>{error}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 p-8">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
          <h1 className="text-4xl font-bold text-white flex items-center gap-3">
            <Camera className="w-10 h-10" />
            Advanced AI Vision Analytics
          </h1>
          <div className="mt-2 flex gap-4">
            <div className="text-green-400 flex items-center gap-2">
              <Zap className="w-4 h-4" />
              {fps} FPS
            </div>
            <div className="text-blue-400 flex items-center gap-2">
              <Clock className="w-4 h-4" />
              {processingTime.toFixed(1)}ms/frame
            </div>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="space-y-8">
            <div className="relative bg-gray-800 rounded-lg overflow-hidden shadow-xl">
              <video
                ref={videoRef}
                className="w-full"
                style={{ maxHeight: '600px' }}
                autoPlay
                playsInline
                muted
              />
              <canvas
                ref={canvasRef}
                className="absolute top-0 left-0 w-full h-full"
              />
            </div>

            <div className="bg-gray-800 rounded-lg p-6 shadow-xl">
              <div className="flex items-center gap-2 mb-4">
                <BrainCircuit className="text-purple-400" />
                <h2 className="text-xl font-semibold text-white">Image Classification</h2>
              </div>
              <div className="space-y-2">
                {classifications.map((classification, index) => (
                  <div
                    key={index}
                    className="bg-gray-700 rounded-lg p-3 flex justify-between items-center"
                  >
                    <span className="text-white">{classification.className}</span>
                    <span className="text-purple-400 font-semibold">
                      {(classification.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="space-y-8">
            {/* Key Metrics */}
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-800 rounded-lg p-6 shadow-xl">
                <div className="flex items-center gap-2 text-cyan-400">
                  <Target className="w-6 h-6" />
                  <h3 className="text-lg font-semibold">Total Detections</h3>
                </div>
                <p className="text-3xl font-bold text-white mt-2">{totalDetections}</p>
                <p className="text-white mt-2">This graph shows the total number of objects detected by the AI model in real-time.</p>
              </div>
              <div className="bg-gray-800 rounded-lg p-6 shadow-xl">
                <div className="flex items-center gap-2 text-pink-400">
                  <Shield className="w-6 h-6" />
                  <h3 className="text-lg font-semibold">Avg Confidence</h3>
                </div>
                <p className="text-3xl font-bold text-white mt-2">
                  {(averageConfidence * 100).toFixed(1)}%
                </p>
                <p className="text-white mt-2">This graph displays the average confidence level of the detections made by the AI model.</p>
              </div>
            </div>

            {/* Real-time Detection Rate */}
            <div className="bg-gray-800 rounded-lg p-6 shadow-xl">
              <div className="flex items-center gap-2 mb-4">
                <Clock className="text-green-400" />
                <h2 className="text-xl font-semibold text-white">Detection Metrics</h2>
              </div>
              <p className="text-white mb-4">This graph shows the number of detections and the average confidence over time, updated in real-time.</p>
              <div className="h-[200px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={timeSeriesData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="timestamp"
                      stroke="#fff"
                      tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                    />
                    <YAxis stroke="#fff" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1f2937',
                        border: 'none',
                        borderRadius: '0.5rem',
                        color: '#fff',
                      }}
                      labelFormatter={(value) => new Date(value).toLocaleTimeString()}
                    />
                    <Line
                      type="monotone"
                      dataKey="detections"
                      stroke="#4ade80"
                      name="Detections"
                    />
                    <Line
                      type="monotone"
                      dataKey="avgConfidence"
                      stroke="#f472b6"
                      name="Confidence"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Object Distribution */}
            <div className="bg-gray-800 rounded-lg p-6 shadow-xl">
              <div className="flex items-center gap-2 mb-4">
                <BarChart3 className="text-cyan-400" />
                <h2 className="text-xl font-semibold text-white">Object Distribution</h2>
              </div>
              <p className="text-white mb-4">This bar chart represents the distribution of different object classes detected by the AI model.</p>
              <div className="h-[200px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={stats}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" stroke="#fff" />
                    <YAxis stroke="#fff" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1f2937',
                        border: 'none',
                        borderRadius: '0.5rem',
                        color: '#fff',
                      }}
                    />
                    <Bar dataKey="count">
                      {stats.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Confidence Distribution */}
            <div className="bg-gray-800 rounded-lg p-6 shadow-xl">
              <div className="flex items-center gap-2 mb-4">
                <Layers className="text-yellow-400" />
                <h2 className="text-xl font-semibold text-white">Confidence Distribution</h2>
              </div>
              <p className="text-white mb-4">This pie chart shows the distribution of confidence levels for the detected objects.</p>
              <div className="h-[200px]">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={confidenceDistribution}
                      dataKey="count"
                      nameKey="range"
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      label
                    >
                      {confidenceDistribution.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1f2937',
                        border: 'none',
                        borderRadius: '0.5rem',
                        color: '#fff',
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-8 bg-gray-800 rounded-lg p-6 shadow-xl">
          <h2 className="text-xl font-semibold text-white mb-4">Current Detections</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {detections.map((detection, index) => (
              <div
                key={index}
                className="bg-gray-700 rounded-lg p-4 text-white shadow-md hover:shadow-lg transition-shadow"
              >
                <div className="font-semibold text-cyan-400">{detection.class}</div>
                <div className="text-gray-300">
                  Confidence: {Math.round(detection.score * 100)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;