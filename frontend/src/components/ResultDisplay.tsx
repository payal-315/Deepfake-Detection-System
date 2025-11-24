'use client'

import { motion, AnimatePresence } from 'framer-motion'
import { CheckCircle, XCircle, AlertTriangle, Clock, Gauge, Brain, Eye } from 'lucide-react'
import { useRef, useEffect, useState } from "react"

interface ResultDisplayProps {
  result: {
    id: string
    filename: string
    file_type: string
    framegrad: Record<string, unknown>
    is_deepfake: boolean
    confidence: number
    processing_time: number
    timestamp: string
    details?: Record<string, unknown>
  } | null
  isProcessing: boolean
}

interface VideoPlayerWithHeatProps {
  videoUrl: string
  perFrame: number[]
  videoDurationSeconds: number | undefined
  API_BASE: string
}

function VideoPlayerWithHeat({ videoUrl, perFrame, videoDurationSeconds, API_BASE }: VideoPlayerWithHeatProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const [bgColor, setBgColor] = useState("bg-green-200")

  useEffect(() => {
    const videoEl = videoRef.current
    if (!videoEl || !videoDurationSeconds) return

    const handleTimeUpdate = () => {
      const currentTime = videoEl.currentTime
      const frameIndex = Math.min(
        Math.floor((currentTime / videoDurationSeconds) * perFrame.length),
        perFrame.length - 1
      )
      const conf = perFrame[frameIndex]

      if (conf > 0.8) setBgColor("bg-red-600")
      else if (conf > 0.7) setBgColor("bg-red-600/60")
      else if (conf > 0.5) setBgColor("bg-red-600/70")
      else if (conf > 0.3) setBgColor("bg-yellow-200")
      else setBgColor("bg-green-300")
    }

    videoEl.addEventListener("timeupdate", handleTimeUpdate)
    return () => videoEl.removeEventListener("timeupdate", handleTimeUpdate)
  }, [perFrame, videoDurationSeconds])

  return (
    <div className={`rounded-xl p-4 transition-colors duration-500 ${bgColor}`}>
      <video
        ref={videoRef}
        src={videoUrl.startsWith("http") ? videoUrl : `${API_BASE}/${videoUrl}`}
        controls
        className="w-full h-[50vh] rounded-lg"
      />
    </div>
  )
}

export default function ResultDisplay({ result, isProcessing }: ResultDisplayProps) {
  if (isProcessing) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="bg-white rounded-2xl p-8 border border-gray-200 text-center"
      >
        <div className="space-y-6">
          {/* Animated processing icon */}
          <div className="relative mx-auto w-24 h-24">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
              className="w-24 h-24 border-4 border-purple-500/30 border-t-purple-500 rounded-full"
            />
            <div className="absolute inset-0 flex items-center justify-center pl-2">
              <Brain className="h-8 w-8 text-purple-400 animate-pulse" />
            </div>
          </div>

          <div>
            <h3 className="text-2xl font-bold text-gray-900 mb-2 mt-6">AI Analysis in Progress</h3>
            <p className="text-gray-700">Our advanced neural networks are analyzing your media...</p>
          </div>

          {/* Processing steps */}
          <div className="space-y-3 mt-6">
            <motion.div
              initial={{ opacity: 0.5 }}
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="flex items-center space-x-3 text-purple-700"
            >
              <div className="w-2 h-2 bg-purple-600 rounded-full animate-pulse" />
              <span>Preprocessing media file...</span>
            </motion.div>
            <motion.div
              initial={{ opacity: 0.5 }}
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
              className="flex items-center space-x-3 text-blue-700"
            >
              <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" />
              <span>Running AI detection models...</span>
            </motion.div>
            <motion.div
              initial={{ opacity: 0.5 }}
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 2, repeat: Infinity, delay: 1 }}
              className="flex items-center space-x-3 text-green-700"
            >
              <div className="w-2 h-2 bg-green-600 rounded-full animate-pulse" />
              <span>Analyzing results...</span>
            </motion.div>
          </div>
        </div>
      </motion.div>
    )
  }

  if (!result) return null

  const confidencePercentage = Math.round(result.confidence * 100)
  const isHighConfidence = result.confidence > 0.8
  const isMediumConfidence = result.confidence > 0.6 && result.confidence <= 0.8

  const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'
  const gradcamPath: string | undefined = typeof result.details?.gradcam_url === 'string' ? result.details.gradcam_url : undefined
  const gradcamface: string | undefined = typeof result.details?.facegrad === 'string' ? result.details.facegrad : undefined

  const gradcamUrl = gradcamPath
    ? (gradcamPath.startsWith('http') ? gradcamPath : `${API_BASE}${gradcamPath}`)
    : null

  const gradface = gradcamface
    ? (gradcamface.startsWith('http') ? gradcamface : `${API_BASE}${gradcamface}`)
    : null

  const perFrameRaw: unknown = result.file_type === 'video'
    ? (result.details?.frame_scores
      ?? result.details?.per_frame_scores
      ?? result.details?.scores
      ?? result.details?.predictions)
    : null

  const perFrame: number[] | null = Array.isArray(perFrameRaw)
    ? (perFrameRaw as number[]).map((v) => {
      const n = Number(v)
      if (Number.isNaN(n)) return 0
      if (n < 0) return 0
      if (n > 1) return 1
      return n
    })
    : null

  const videoFps: number | undefined = typeof result.details?.fps === 'number' ? result.details?.fps : undefined
  const videoDurationSeconds: number | undefined = typeof result.details?.duration_seconds === 'number' ? result.details?.duration_seconds : undefined

  const getResultIcon = () => {
    if (result.is_deepfake) {
      return <XCircle className="h-12 w-12 text-red-400" />
    }
    return <CheckCircle className="h-12 w-12 text-green-400" />
  }

  const getResultColor = () => {
    if (result.is_deepfake) {
      return 'from-red-500/20 to-red-600/20 border-red-500/30'
    }
    return 'from-green-500/20 to-green-600/20 border-green-500/30'
  }

  const getConfidenceColor = () => {
    if (!result.is_deepfake){
      if (isHighConfidence) return 'text-green-600'
      if (isMediumConfidence) return 'text-green-500'
      return 'text-green-200'
    }
    if (isHighConfidence) return 'text-red-600'
    if (isMediumConfidence) return 'text-red-500'
    return 'text-red-400'

  }

  const getConfidenceText = () => {
    if (isHighConfidence) return 'High Confidence'
    if (isMediumConfidence) return 'Medium Confidence'
    return 'Low Confidence'
  }

  const getConfidenceBarColor = () => {
    if (result.is_deepfake){
    if (isHighConfidence) return 'bg-gradient-to-r from-red-600 to-red-500'
    if (isMediumConfidence) return 'bg-gradient-to-r from-red-500 to-red-400'
    return 'bg-gradient-to-r from-yellow-500 to-yellow-400'
    }
    if (isHighConfidence) return 'bg-gradient-to-r from-green-600 to-green-500'
    if (isMediumConfidence) return 'bg-gradient-to-r from-green-500 to-green-400'
    return 'bg-gradient-to-r from-yellow-500 to-yellow-400'
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 20, scale: 0.9 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.5 }}
        className={`bg-white rounded-2xl p-8 border ${result.is_deepfake ? 'border-red-300' : 'border-green-300'}`}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center space-x-4">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
            >
              {getResultIcon()}
            </motion.div>
            <div>
              <motion.h3
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 }}
                className="text-3xl font-bold text-gray-900"
              >
                {result.is_deepfake ? 'Deepfake Detected' : 'Authentic Content'}
              </motion.h3>
              <motion.p
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4 }}
                className="text-gray-600"
              >
                {result.filename}
              </motion.p>
            </div>
          </div>
          <motion.div
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.5 }}
            className="text-right"
          >
            <div className="text-sm text-gray-600">Processed in</div>
            <div className="text-2xl font-bold text-gray-900">
              {result.processing_time.toFixed(2)}s
            </div>
          </motion.div>
        </div>

        {/* Confidence Score */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="mb-8"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <Gauge className="h-5 w-5 text-gray-900" />
              <span className="text-lg font-semibold text-gray-900">{result.file_type==='video' && 'Mean '}Confidence Score</span>
            </div>
            <span className={`text-lg font-bold ${getConfidenceColor()}`}>
              {confidencePercentage}% ({getConfidenceText()})
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${confidencePercentage}%` }}
              transition={{ duration: 1.5, delay: 0.7, ease: "easeOut" }}
              className={`h-4 rounded-full ${getConfidenceBarColor()} relative`}
            >
              <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
            </motion.div>
          </div>
        </motion.div>


        {perFrame && perFrame.length > 1 && (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay: 0.6 }}
    className="mb-8"
  >
    <div className="flex items-center justify-between mb-4">
      <div className="flex items-center space-x-2">
        <Gauge className="h-5 w-5 text-gray-900" />
        <span className="text-lg font-semibold text-gray-900">
          Highest Confidence Score
        </span>
      </div>

      {/* ✅ Take highest confidence */}
      {(() => {
        const highestConfidence = Math.max(...perFrame) * 100
        return (
          <span className={`text-lg font-bold ${getConfidenceColor()}`}>
            {highestConfidence.toFixed(1)}% ({getConfidenceText()})
          </span>
        )
      })()}
    </div>

    <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
      {(() => {
        const highestConfidence = Math.max(...perFrame) * 100
        return (
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${highestConfidence}%` }}
            transition={{ duration: 1.5, delay: 0.7, ease: "easeOut" }}
            className={`h-4 rounded-full ${getConfidenceBarColor()} relative`}
          >
            <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
          </motion.div>
        )
      })()}
    </div>
  </motion.div>
)}



        {/* Details Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8"
        >
          <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
            <div className="flex items-center space-x-3">
              <Clock className="h-6 w-6 text-blue-600" />
              <div>
                <div className="text-sm text-gray-600">Processing Time</div>
                <div className="font-bold text-gray-900">{result.processing_time.toFixed(2)} seconds</div>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
            <div className="flex items-center space-x-3">
              <Eye className="h-6 w-6 text-purple-600" />
              <div>
                <div className="text-sm text-gray-600">File Type</div>
                <div className="font-bold text-gray-900 capitalize">{result.file_type}</div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Per-frame predictions graph (videos) */}
        {perFrame && perFrame.length > 1 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1 }}
            className="mb-8 bg-white rounded-2xl p-6 border border-gray-200"
          >
            <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center space-x-2">
              <span>Per-frame Prediction</span>
              <span className="text-xs font-normal text-gray-500">(0 = authentic, 1 = deepfake)</span>
            </h4>
            {/* Simple responsive SVG line chart with time on X axis */}
            <div className="w-full">
              {(() => {
                const width = 1000
                const height = 250
                const paddingX = 32
                const paddingY = 16
                const usableWidth = width - paddingX * 2
                const usableHeight = height - paddingY * 2
                const n = perFrame.length
                // Map index to time, if duration provided; else use frame index scale
                const xFor = (i: number) => paddingX + (i / (n - 1)) * usableWidth
                const yFor = (v: number) => paddingY + (1 - v) * usableHeight
                // Build color-coded line segments by threshold buckets
                type Bucket = 'red' | 'yellow' | 'green'
                const getBucket = (v: number): Bucket => (v > 0.6 ? 'red' : v > 0.35 ? 'yellow' : 'green')
                const paths: Record<Bucket, string[]> = { red : [], yellow : [], green : [] }
                const areaPaths: Record<Bucket, string[]> = { red: [], yellow: [], green: [] }
                let currentBucket: Bucket | null = null
                let segStart = 0
                for (let i = 0; i < n; i++) {
                  const v = perFrame[i]
                  const bucket = getBucket(v)
                  const cmd = `${i === 0 ? 'M' : 'L'} ${xFor(i).toFixed(2)} ${yFor(v).toFixed(2)}`
                  paths['red'].push(cmd)
                  if (currentBucket === null) {
                    currentBucket = bucket
                    segStart = 0
                  } else if (bucket !== currentBucket) {
                    // Close previous segment area [segStart..i-1]
                    const startX = xFor(segStart).toFixed(2)
                    const startY = yFor(perFrame[segStart]).toFixed(2)
                    const segCmds: string[] = []
                    segCmds.push(`M ${startX} ${startY}`)
                    for (let j = segStart + 1; j <= i - 1; j++) {
                      segCmds.push(`L ${xFor(j).toFixed(2)} ${yFor(perFrame[j]).toFixed(2)}`)
                    }
                    // down to baseline and back to start
                    segCmds.push(`L ${xFor(i - 1).toFixed(2)} ${yFor(0).toFixed(2)}`)
                    segCmds.push(`L ${xFor(segStart).toFixed(2)} ${yFor(0).toFixed(2)}`)
                    segCmds.push('Z')
                    areaPaths[currentBucket].push(segCmds.join(' '))
                    currentBucket = bucket
                    segStart = i
                  }
                }
                // Close final segment area [segStart..n-1]
                if (currentBucket !== null) {
                  const startX = xFor(segStart).toFixed(2)
                  const startY = yFor(perFrame[segStart]).toFixed(2)
                  const segCmds: string[] = []
                  segCmds.push(`M ${startX} ${startY}`)
                  for (let j = segStart + 1; j <= n - 1; j++) {
                    segCmds.push(`L ${xFor(j).toFixed(2)} ${yFor(perFrame[j]).toFixed(2)}`)
                  }
                  segCmds.push(`L ${xFor(n - 1).toFixed(2)} ${yFor(0).toFixed(2)}`)
                  segCmds.push(`L ${xFor(segStart).toFixed(2)} ${yFor(0).toFixed(2)}`)
                  segCmds.push('Z')
                  areaPaths[currentBucket].push(segCmds.join(' '))
                }
                return (
                  <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-52">
                    {/* Background grid */}
                    <rect x="0" y="0" width={width} height={height} fill="white" />
                    {/* Y-axis ticks */}
                    {[0, 0.25, 0.5, 0.75, 1].map((t) => (
                      <g key={t}>
                        <line
                          x1={paddingX}
                          x2={width - paddingX}
                          y1={yFor(t)}
                          y2={yFor(t)}
                          stroke="#e5e7eb"
                          strokeWidth={1}
                        />
                        <text x={8} y={yFor(t) + 4} fontSize={10} fill="#6b7280">{t.toFixed(2)}</text>
                      </g>
                    ))}
                    {/* X-axis ticks (time) */}
                    {(() => {
                      const ticks: number[] = []
                      const maxTicks = 8
                      if (videoDurationSeconds && videoDurationSeconds > 0) {
                        const step = Math.max(1, Math.round(videoDurationSeconds / maxTicks))
                        for (let s = 0; s <= Math.floor(videoDurationSeconds); s += step) {
                          ticks.push(s)
                        }
                      } else {
                        // fallback to frame index ticks
                        const step = Math.max(1, Math.floor(n / maxTicks))
                        for (let i = 0; i < n; i += step) {
                          ticks.push(i)
                        }
                      }
                      return (
                        <g>
                          {ticks.map((t, idx) => {
                            const x = videoDurationSeconds && videoDurationSeconds > 0
                              ? paddingX + (t / videoDurationSeconds) * usableWidth
                              : paddingX + ((t as number) / (n - 1)) * usableWidth
                            const label = videoDurationSeconds && videoDurationSeconds > 0
                              ? `${t}s`
                              : `${t}`
                            return (
                              <g key={`xtick-${idx}`}>
                                <line x1={x} x2={x} y1={height - paddingY} y2={height - paddingY + 4} stroke="#9ca3af" strokeWidth={1} />
                                <text x={x} y={height - 2} fontSize={10} fill="#6b7280" textAnchor="middle">{label}</text>
                              </g>
                            )
                          })}
                        </g>
                      )
                    })()}
                    {/* Color-coded areas under line */}
                    {areaPaths.green.map((d, idx) => (
                      <path key={`ag-${idx}`} d={d} fill="#22c55e33" stroke="none" />
                    ))}
                    {areaPaths.yellow.map((d, idx) => (
                      <path key={`ay-${idx}`} d={d} fill="#f59e0b33" stroke="none" />
                    ))}
                    {areaPaths.red.map((d, idx) => (
                      <path key={`ar-${idx}`} d={d} fill="#ef444433" stroke="none" />
                    ))}
                    {/* Color-coded line segments */}
                    {result.is_deepfake ?
                      <path d={paths.red.join(' ')} fill="none" stroke="#ef4444" strokeWidth={2} />

                      :
                      <>
                        <path d={paths.red.join(' ')} fill="none" stroke="#16a34a" strokeWidth={2} />
                      </>
                    }
                  </svg>
                )
              })()}
            </div>
            <div className="mt-2 text-xs text-gray-600">Frames: {perFrame.length}</div>
          </motion.div>
        )}

        {result.file_type === "video" && perFrame && perFrame.length > 1 && typeof result.details?.video_url === 'string' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.2 }}
            className="mb-8 rounded-2xl p-4"
            style={{ backgroundColor: "#f9fafb" }}
          >
            <h4 className="text-lg font-semibold text-gray-900 mb-4">
              Video Playback with Confidence Heat Background
            </h4>

            <VideoPlayerWithHeat
              videoUrl={typeof result.details.video_url === 'string' ? result.details.video_url : ''}
              perFrame={perFrame}
              videoDurationSeconds={videoDurationSeconds}
              API_BASE={API_BASE}
            />
          </motion.div>
        )}
        {/* Additional Details */}
        {result.details && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1 }}
            className="border-t border-gray-200 pt-6"
          >
            <h4 className="text-xl font-semibold text-gray-900 mb-6 flex items-center space-x-2">
              <Brain className="h-6 w-6" />
              <span>Analysis Details</span>
            </h4>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {result.file_type === 'image' && result.details.face_detected !== undefined && (
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  className="bg-blue-50 border border-blue-200 rounded-xl p-4"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-blue-700 font-medium">Face Detected</span>
                    <span className={`font-bold ${result.details.face_detected ? 'text-green-600' : 'text-red-600'}`}>
                      {result.details.face_detected ? 'Yes' : 'No'}
                    </span>
                  </div>
                </motion.div>
              )}

              {result.details.quality_score && (
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  className="bg-green-50 border border-green-200 rounded-xl p-4"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-green-700 font-medium">Quality Score</span>
                    <span className="font-bold text-green-700">
                      {(result.details.quality_score * 100).toFixed(1)}%
                    </span>
                  </div>
                </motion.div>
              )}

              {result.details.artifacts_found !== undefined && (
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  className="bg-yellow-50 border border-yellow-200 rounded-xl p-4"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-yellow-700 font-medium">Artifacts Found</span>
                    <span className="font-bold text-yellow-700">
                      {result.details.artifacts_found}
                    </span>
                  </div>
                </motion.div>
              )}

              {result.details.frames_analyzed && (
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  className="bg-purple-50 border border-purple-200 rounded-xl p-4"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-purple-700 font-medium">Frames Analyzed</span>
                    <span className="font-bold text-purple-700">
                      {result.details.frames_analyzed}
                    </span>
                  </div>
                </motion.div>
              )}

              {result.details.temporal_consistency && (
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  className="bg-indigo-50 border border-indigo-200 rounded-xl p-4"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-indigo-700 font-medium">Temporal Consistency</span>
                    <span className="font-bold text-indigo-700">
                      {(result.details.temporal_consistency * 100).toFixed(1)}%
                    </span>
                  </div>
                </motion.div>
              )}

              {result.details.audio_sync_score && (
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  className="bg-pink-50 border border-pink-200 rounded-xl p-4"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-pink-700 font-medium">Audio Sync Score</span>
                    <span className="font-bold text-pink-700">
                      {(result.details.audio_sync_score * 100).toFixed(1)}%
                    </span>
                  </div>
                </motion.div>
              )}

            </div>

            {result.details.image_details && (
              <div className="mt-6">
                <h5 className="text-gray-900 font-semibold mb-3">Feature Visualization</h5>
                <div className="bg-gray-50 border border-gray-200 rounded-xl overflow-hidden">
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-2 p-2">
                    <div className="flex flex-col items-center justify-center">
                      <p>Noise</p>
                      <img
                        src={result.details.image_details.noise}
                        alt={`noise`}
                        className="w-full max-h-[240px] object-contain bg-gray-200 rounded-lg"
                      />
                    </div>
                    <div className="flex flex-col items-center justify-center">
                      <p>ELA</p>
                      <img
                        src={result.details.image_details.ela}
                        alt={`noise`}
                        className="w-full max-h-[240px] object-contain bg-gray-200 rounded-lg"
                      />
                    </div>
                    <div className="flex flex-col items-center justify-center">
                      <p>Frequency</p>
                      <img
                        src={result.details.image_details.freq}
                        alt={`noise`}
                        className="w-full max-h-[240px] object-contain bg-gray-200 rounded-lg"
                      />
                    </div>
                  </div>

                </div>
              </div>
            )}
            {gradcamUrl && (
              <div className="mt-6">
                <h5 className="text-gray-900 font-semibold mb-3">Grad-CAM Visualization</h5>
                <div className="bg-gray-50 border border-gray-200 rounded-xl overflow-hidden">
                  {result?.file_type === "video" ? (
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2 p-2">
                      {result.framegrad?.map((frame: string, idx: number) => (
                        <div key={idx} className="flex flex-col items-center justify-center">
                          <p>
                            {videoFps
                              ? `${(idx * (15 / videoFps)).toFixed(1)}s | Frame ${idx}`
                              : `Frame ${idx}`}

                          </p>
                          <img

                            src={frame.startsWith("http") ? frame : `${API_BASE}/${frame}`}
                            alt={`Grad-CAM frame ${idx}`}
                            className="w-full max-h-[240px] object-contain bg-gray-200 rounded-lg"
                          />
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className='flex justify-center items-center w-full'>
                      <img
                        src={gradcamUrl}
                        alt="Grad-CAM"
                        className="w-full m-2 max-h-[480px] object-contain bg-gray-200"
                      />
                      {gradface &&

                        <img
                          src={gradface}
                          alt="Grad-CAM-face"
                          className="w-full m-2 max-h-[480px] object-contain bg-gray-200"
                        />
                      }
                    </div>
                  )}
                </div>
                <p className="text-xs text-gray-600 mt-2">Heatmap highlighting regions most influential to the model&apos;s prediction.</p>
              </div>
            )}
          </motion.div>
        )}
{result.details?.pdf_path && (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay: 1.2 }}
    className="border-t border-gray-200 pt-6"
  >
    <h4 className="text-xl font-semibold text-gray-900 mb-4 flex items-center space-x-2">
      <span>Forensic Report</span>
    </h4>
    <a
      href={result.details.pdf_path.startsWith("http") ? result.details.pdf_path : `${API_BASE}${result.details.pdf_path}`}
      download
      target="_blank"
      rel="noopener noreferrer"
      className="inline-block bg-blue-600 text-white font-medium px-4 py-2 rounded-lg hover:bg-blue-700 transition"
    >
      Download PDF Report
    </a>
  </motion.div>
)}
        {/* Warning/Info Banner */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.2 }}
          className={`mt-8 p-6 rounded-xl ${result.is_deepfake
            ? 'bg-red-500/20 border border-red-500/30'
            : 'bg-green-500/20 border border-green-500/30'
            }`}
        >
          <div className="flex items-start space-x-4">
            <AlertTriangle className={`h-6 w-6 mt-0.5 ${result.is_deepfake ? 'text-red-400' : 'text-green-400'
              }`} />
            <div>
              <h5 className={`font-bold text-lg ${result.is_deepfake ? 'text-red-300' : 'text-green-300'
                }`}>
                {result.is_deepfake ? '⚠️ Deepfake Detected' : '✅ Content Appears Authentic'}
              </h5>
              <p className={`mt-2 ${result.is_deepfake ? 'text-red-950' : 'text-green-950'
                }`}>
                {result.is_deepfake
                  ? 'This content shows signs of being artificially generated or manipulated. Please verify the source and context before sharing or making decisions based on this content.'
                  : 'No significant signs of manipulation were detected. However, always verify content from trusted sources and use multiple verification methods for important decisions.'
                }
              </p>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}
