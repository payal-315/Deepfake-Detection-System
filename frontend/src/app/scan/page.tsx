"use client"
export const dynamic = "force-dynamic";

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import FileUpload from '@/components/FileUpload'
import ResultDisplay from '@/components/ResultDisplay'
import NavBar from '../../components/NavBar'
import { useAuth } from '../../components/AuthContext'
import { useRouter } from 'next/navigation'

interface DetectionResult {
  id: string
  filename: string
  file_type: string
  framegrad: Record<string, unknown>
  is_deepfake: boolean
  confidence: number
  processing_time: number
  timestamp: string
  details?: Record<string, unknown>
}

interface HistoryItem {
  _id: string
  filename: string
  file_type: string
  is_deepfake: boolean
  confidence: number
  processing_time: number
  timestamp: string
  details?: Record<string, unknown>
}

function Waveform({ file, color = '#7c3aed' }: { file: File, color?: string }) {
  const [points, setPoints] = useState<number[]>([])

  useEffect(() => {
    let isCancelled = false
    const run = async () => {
      try {
        const arrayBuffer = await file.arrayBuffer()
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)()
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0))
        const channelData = audioBuffer.getChannelData(0)
        const samples = 1000
        const blockSize = Math.floor(channelData.length / samples) || 1
        const newPoints: number[] = []
        for (let i = 0; i < samples; i++) {
          const start = i * blockSize
          let sum = 0
          for (let j = 0; j < blockSize && start + j < channelData.length; j++) {
            sum += Math.abs(channelData[start + j]) * 2
          }
          newPoints.push(sum / blockSize)
        }
        if (!isCancelled) setPoints(newPoints)
        audioCtx.close()
      } catch {
        if (!isCancelled) setPoints([])
      }
    }
    run()
    return () => { isCancelled = true }
  }, [file])

  if (points.length === 0) {
    return <div className="w-full h-24 bg-gray-100 rounded-lg border border-gray-200 flex items-center justify-center text-xs text-gray-500">Waveform unavailable</div>
  }

  const width = 1000
  const height = 120
  const padding = 8
  const usableHeight = height - padding * 2
  const xFor = (i: number) => padding + (i / (points.length - 1)) * (width - padding * 2)
  const yFor = (v: number) => padding + (1 - Math.min(1, v)) * usableHeight
  const path = points.map((v, i) => `${i === 0 ? 'M' : 'L'} ${xFor(i).toFixed(2)} ${yFor(v).toFixed(2)}`).join(' ')

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-24">
      <rect x="0" y="0" width={width} height={height} fill="#ffffff" />
      <path d={path} fill="none" stroke={color} strokeWidth={2} />
    </svg>
  )
}

export default function ScanPage() {
  const { user, token, loading } = useAuth()
  const router = useRouter()

  // Media (image/video) tab state
  const [mediaResult, setMediaResult] = useState<DetectionResult | null>(null)
  const [isProcessingMedia, setIsProcessingMedia] = useState(false)
  // Video-Audio tab state
  const [vaResult, setVaResult] = useState<DetectionResult | null>(null)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [fvaResult, setFVaResult] = useState<DetectionResult | null>(null)
  const [isProcessingVa, setIsProcessingVa] = useState(false)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [fisProcessingVa, setFIsProcessingVa] = useState(false)
  const [refAudio, setRefAudio] = useState<File | null>(null)
  const [testAudio, setTestAudio] = useState<File | null>(null)
  const [audioResult, setAudioResult] = useState<{ similarity: number, verdict: string, probability: number,pdf_path : string } | null>(null)
  const [activeTab, setActiveTab] = useState<'media' | 'audio' | 'video-audio'>('media')
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [showHistory, setShowHistory] = useState(false)
  const [audioloading, setAudioLoading] = useState(false);
  const [facedetec, setFacedetec] = useState(false);


  useEffect(() => {
    if (!loading && !user) {
      router.push('/auth')
    }
  }, [user, loading, router])

  const loadHistory = async () => {
    if (!token) return

    try {
      const res = await fetch('http://localhost:8000/history', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
      if (!res.ok) return
      const data = await res.json()
      setHistory(Array.isArray(data?.detections) ? data.detections : [])
    } catch { }
  }

  useEffect(() => {
    if (user && token) {
      loadHistory()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user, token])

  // Media handlers
  const handleMediaDetectionComplete = (data: unknown) => {
    setMediaResult(data as DetectionResult)
    setIsProcessingMedia(false)
    loadHistory()
  }

  const handleMediaDetectionStart = () => {
    setIsProcessingMedia(true)
    setMediaResult(null)
  }

  // Video-Audio handlers
  const handleVaDetectionComplete = (data: unknown) => {
    setVaResult(data as DetectionResult)
    setIsProcessingVa(false)
    loadHistory()
  }

  const handleVaDetectionStart = () => {
    setIsProcessingVa(true)
    setVaResult(null)
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const fhandleVaDetectionComplete = (data: unknown) => {
    setFVaResult(data as DetectionResult)
    setFIsProcessingVa(false)
    loadHistory()
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const fhandleVaDetectionStart = () => {
    setFIsProcessingVa(true)
    setFVaResult(null)
  }


  if (loading) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-red-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    )
  }

  if (!user) {
    return null // Will redirect to /auth
  }

  return (
    <div className="min-h-screen bg-white">
      <NavBar />

      <div className="relative bg-transparent pt-16">
        <div className="absolute inset-0 overflow-hidden pointer-events-none" />

        <div className="relative mb-8 z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="mb-6 flex items-center justify-between">
            {/* Tabs */}
            <div className="inline-flex items-center  gap-2 rounded-xl border border-gray-200 p-1 bg-white">
              <button
                onClick={() => setActiveTab('media')}
                className={`${activeTab === 'media' ? 'bg-red-600 text-white' : 'bg-white text-gray-700 hover:bg-gray-50'} px-4 py-2 rounded-lg text-sm font-medium`}
              >
                Image / Video
              </button>
              <button
                onClick={() => setActiveTab('audio')}
                className={`${activeTab === 'audio' ? 'bg-red-600 text-white' : 'bg-white text-gray-700 hover:bg-gray-50'} px-4 py-2 rounded-lg text-sm font-medium`}
              >
                Audio
              </button>
              <button
                onClick={() => setActiveTab('video-audio')}
                className={`${activeTab === 'video-audio' ? 'bg-red-600 text-white' : 'bg-white text-gray-700 hover:bg-gray-50'} px-4 py-2 rounded-lg text-sm font-medium`}
              >
                Video Audio
              </button>
            </div>
            {/* <button
              onClick={() => setShowHistory(!showHistory)}
              className="px-3 py-2 rounded-lg border border-gray-200 bg-white text-gray-700 hover:bg-gray-50 text-sm"
            >
              {showHistory ? 'Hide History' : 'Show History'}
            </button> */}
          </div>
          <div className={`grid ${showHistory ? 'grid-cols-1 lg:grid-cols-3' : 'grid-cols-1'} gap-8 items-start`}>
            <div className={`${showHistory ? 'lg:col-span-2' : 'lg:col-span-3'}`}>

              {activeTab === 'media' && (
                <>
                  <motion.h1
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                    className="text-3xl md:text-5xl font-bold text-gray-900 mb-2"
                  >
                    Scan your media
                  </motion.h1>
                  <p className="text-gray-700 mb-12">Upload an image or video to detect potential deepfakes with AI.</p>

                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                  >
                    <FileUpload
                      onDetectionStart={handleMediaDetectionStart}
                      onDetectionComplete={handleMediaDetectionComplete}
                      isProcessing={isProcessingMedia}
                      token={token}
                      facedetect={facedetec}
                    />
                  </motion.div>

                  {(mediaResult || isProcessingMedia) && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.5 }}
                      className="pt-8"
                    >
                      <ResultDisplay result={mediaResult} isProcessing={isProcessingMedia} />
                    </motion.div>
                  )}
                </>
              )}


              {activeTab === 'audio' && (
                <div className="mt-2">
                  <motion.h2
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                    className="text-2xl md:text-3xl font-bold text-gray-900 mb-4"
                  >
                    Reference Scan
                  </motion.h2>
                  <p className="text-gray-700 mb-6">Upload a reference audio and a test audio to compare.</p>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="bg-white rounded-2xl p-6 border border-gray-200">
                      <label className="block text-sm font-medium text-gray-700 mb-2">Reference Audio</label>
                      <input
                        type="file"
                        accept="audio/*"
                        onChange={(e) => setRefAudio(e.target.files?.[0] || null)}
                        className="block w-full text-sm text-gray-700 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-red-50 file:text-red-700 hover:file:bg-red-100"
                      />
                      {refAudio && (
                        <>
                          <audio key={refAudio.name + refAudio.lastModified} controls className="w-full mt-3">
                            <source src={URL.createObjectURL(refAudio)} />
                          </audio>
                          <div className="mt-3">
                            <Waveform key={refAudio.name + refAudio.lastModified} file={refAudio} color="#dc2626" />
                          </div>
                        </>
                      )}


                    </div>
                    <div className="bg-white rounded-2xl p-6 border border-gray-200">
                      <label className="block text-sm font-medium text-gray-700 mb-2">Test Audio</label>
                      <input
                        type="file"
                        accept="audio/*"
                        onChange={(e) => setTestAudio(e.target.files?.[0] || null)}
                        className="block w-full text-sm text-gray-700 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-red-50 file:text-red-700 hover:file:bg-red-100"
                      />

                      {testAudio && (
                        <>
                          <audio key={testAudio.name + testAudio.lastModified} controls className="w-full mt-3">
                            <source src={URL.createObjectURL(testAudio)} />
                          </audio>
                          <div className="mt-3">
                            <Waveform key={testAudio.name + testAudio.lastModified} file={testAudio} color="#ef4444" />
                          </div>
                        </>
                      )}
                    </div>
                  </div>

                  <div className="mt-4 flex items-center gap-3">
                    <button
                      onClick={async () => {
                        setAudioLoading(true)
                        if (!testAudio || !token) return
                        try {
                          const fd = new FormData()
                          if (refAudio) {
                            fd.append('reference_audio', refAudio)
                          }
                          fd.append('test_audio', testAudio)
                          const res = await fetch('http://localhost:8000/detect/audio/reference', {
                            method: 'POST',
                            headers: {
                              'Authorization': `Bearer ${token}`
                            },
                            body: fd,
                          })
                          if (!res.ok) throw new Error('Request failed')
                          const data = await res.json()
                          setAudioResult({ similarity: data.similarity, verdict: data.verdict, probability: data.probability,pdf_path: data.pdf_path })
                          loadHistory()
                        } catch {
                          setAudioResult(null)
                        } finally {
                          setAudioLoading(false)
                        }
                      }}
                      className="px-5 py-2.5 rounded-xl bg-red-600 text-white hover:bg-red-700"
                    >
                      Compare Audio
                    </button>
                    <button
                      onClick={() => { setRefAudio(null); setTestAudio(null); setAudioResult(null) }}
                      className="px-4 py-2 rounded-xl bg-gray-100 text-gray-700 hover:bg-gray-200"
                    >
                      Reset
                    </button>
                  </div>

                  {audioloading && (
                    <div className="mt-6 bg-yellow-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl flex items-center gap-3">
                      <svg className="animate-spin h-6 w-6 text-red-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"></path>
                      </svg>
                      <span>
                        Comparing audio, please wait...
                      </span>
                    </div>
                  )}
                  {audioResult && !audioloading && (
                    <div className="mt-12 bg-gradient-to-br from-blue-50 via-white to-green-50 rounded-3xl p-8 shadow-lg border border-gray-100 space-y-8">
                      {/* Header */}
                      <div className="flex items-center justify-between my-6">
                        <div className="flex items-center gap-3">
                          <span className={`inline-flex items-center justify-center rounded-full h-10 w-10 ${audioResult.verdict === 'match' || audioResult.verdict === 'real'
                              ? 'bg-green-100 text-green-600'
                              : 'bg-red-100 text-red-600'
                            }`}>
                            {audioResult.verdict === 'match' ? (
                              <svg className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                              </svg>
                            ) : (
                              <svg className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                              </svg>
                            )}
                          </span>
                          <h2 className="text-2xl font-bold text-gray-900">Audio Verification Result</h2>
                        </div>
                        <span
                          className={`px-4 py-1.5 text-base font-semibold rounded-full shadow-sm transition-colors duration-300 ${audioResult.verdict === 'match' || audioResult.verdict === 'real'
                              ? 'bg-green-100 text-green-800 border border-green-200'
                              : 'bg-red-100 text-red-800 border border-red-200'
                            }`}
                        >
                          {audioResult.verdict.toUpperCase()}
                        </span>
                      </div>

                      {/* Similarity */}
                      {audioResult.similarity ?
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-gray-700 font-medium flex items-center gap-2">
                              <svg className="h-5 w-5 text-blue-400" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M3 15a4 4 0 004 4h10a4 4 0 004-4V9a4 4 0 00-4-4H7a4 4 0 00-4 4v6z" />
                              </svg>
                              Similarity
                            </span>
                            <span className="font-bold text-gray-900 text-lg">
                              {(audioResult.similarity * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-3 relative overflow-hidden">
                            <div
                              className={`h-5 rounded-full transition-all duration-700 ${audioResult.similarity > 0.7
                                  ? 'bg-gradient-to-r from-green-400 to-green-600'
                                  : audioResult.similarity > 0.4
                                    ? 'bg-gradient-to-r from-yellow-300 to-yellow-500'
                                    : 'bg-gradient-to-r from-red-400 to-red-600'
                                }`}
                              style={{ width: `${(audioResult.similarity * 100).toFixed(1)}%` }}
                            />
                            <div className="absolute inset-0 flex items-center justify-center text-xs text-gray-500 font-medium">
                              {audioResult.similarity > 0.7
                                ? "High"
                                : audioResult.similarity > 0.4
                                  ? "Medium"
                                  : "Low"}
                            </div>
                          </div>
                        </div>
                        :
                        <div className="text-gray-700 font-medium flex items-center gap-2">
                          No reference audio uploaded
                        </div>
                      }


                      {/* Deepfake Probability */}
                      <div className='mt-4'>
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-gray-700 font-medium flex items-center gap-2">
                            <svg className="h-5 w-5 text-purple-400" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" fill="none" />
                              <path strokeLinecap="round" strokeLinejoin="round" d="M8 12l2 2 4-4" />
                            </svg>
                            Deepfake Probability
                          </span>
                          <span
                            className={`font-bold text-lg ${audioResult.probability > 0.5
                                ? 'text-red-700'
                                : 'text-green-700'
                              }`}
                          >
                            {(audioResult.probability * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-3 relative overflow-hidden">
                          <div
                            className={`h-3 rounded-full transition-all duration-700 bg-gradient-to-r from-red-400 to-red-600
                              `}
                            style={{ width: `${(audioResult.probability * 100).toFixed(1)}%` }}
                          />
                          <div className="absolute inset-0 flex items-center justify-center text-xs text-gray-500 font-medium">
                          
                          </div>
                        </div>
                      </div>

                      {/* Final Verdict */}
                      <div className="text-center pt-6 border-t border-gray-100">
                        <p className="text-2xl font-bold text-gray-800 flex items-center justify-center gap-2">
                          <span>Final Verdict:</span>
                          <span
                            className={`ml-2 px-4 py-1.5 rounded-full text-lg font-semibold shadow-sm transition-colors duration-300 ${audioResult.verdict === 'match' || audioResult.verdict === 'real'
                                ? 'bg-green-100 text-green-700 border border-green-200'
                                : 'bg-red-100 text-red-700 border border-red-200'
                              }`}
                          >
                            {audioResult.verdict === 'match' || audioResult.verdict === 'real' ? (
                              <>
                                <svg className="inline h-5 w-5 mr-1" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                                </svg>
                                {audioResult.verdict.toUpperCase()}
                              </>
                            ) : (
                              <>
                                <svg className="inline h-5 w-5 mr-1" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                                </svg>
                                Mismatch
                              </>
                            )}
                          </span>
                        </p>
                        <p className="mt-2 text-gray-500 text-sm">
                          {audioResult.similarity

                            ?
                            <>
                              {audioResult.verdict === 'match'
                                ? "The reference and test audio are likely from the same speaker."
                                : "The reference and test audio are likely from different speakers or a deepfake was detected."
                              }
                            </>
                            :
                            <>
                              {audioResult.verdict === 'real'
                                ? "The test audio is likely to be real"
                                : "The test audio is likely to be deepfake"
                              }
                            </>
                          }
                        </p>
                        {audioResult.pdf_path && (
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
      href={audioResult.pdf_path.startsWith("http") ? audioResult.pdf_path : `${process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'}${audioResult.pdf_path}`}
      download
      target="_blank"
      rel="noopener noreferrer"
      className="inline-block bg-blue-600 text-white font-medium px-4 py-2 rounded-lg hover:bg-blue-700 transition"
    >
      Download PDF Report
    </a>
  </motion.div>
)}
                      </div>
                    </div>
                  )}

                </div>
              )}

              {activeTab === 'video-audio' && (
                <>
                  <motion.h1
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                    className="text-3xl md:text-5xl font-bold text-gray-900 mb-2"
                  >
                    Scan your video (audio-aware model)
                  </motion.h1>
                  <p className="text-gray-700 mb-12">Upload a video to analyze, using a model that considers both visual and audio cues.</p>

                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                  >
                    <FileUpload
                      onDetectionStart={handleVaDetectionStart}
                      onDetectionComplete={handleVaDetectionComplete}
                      isProcessing={isProcessingVa}
                      mode="video-only"
                      token={token}
                      facedetect={facedetec}
                    />

                    <div className="flex items-center my-6 bg-red-100 p-4">
                      <input
                        id="face-detect-checkbox"
                        type="checkbox"
                        checked={facedetec}
                        onChange={() => setFacedetec((prev) => !prev)}
                        className="mr-2 accent-red-600"
                        disabled={isProcessingVa}
                        style={{ width: "1.5em", height: "1.5em" }}
                      />
                      <label htmlFor="face-detect-checkbox" className="text-gray-800 text-sm font-bold select-none">
                        Enable Face Detection
                      </label>
                    </div>
                  </motion.div>

                  {(vaResult || isProcessingVa) && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.5 }}
                      className="pt-8"
                    >
                      <ResultDisplay result={vaResult} isProcessing={isProcessingVa} />
                    </motion.div>
                  )}
                  {/* {(fvaResult || fisProcessingVa) && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                  className="pt-8"
                >
                  <ResultDisplay result={fvaResult} isProcessing={fisProcessingVa} />
                </motion.div>
              )} */}
                </>
              )}
            </div>

            {/* Sidebar History */}
            {/* {showHistory && (
              <aside className="lg:col-span-1 ">
                <div className="sticky top-24">
                  <div className="bg-white rounded-2xl border border-gray-200 p-5">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-semibold text-gray-900">Recent Scans</h3>
                      <button onClick={loadHistory} className="text-xs px-2 py-1 rounded-md bg-gray-100 text-gray-700 hover:bg-gray-200">Refresh</button>
                    </div>
                    {history.length === 0 ? (
                      <div className="text-sm text-gray-600">No history yet.</div>
                    ) : (
                      <ul className="space-y-3">
                        {history.map((h) => (
                          <li key={h._id} className="border border-gray-200 rounded-xl p-3 hover:bg-gray-50">
                            <div className="flex items-center justify-between">
                              <div className="text-sm font-medium text-gray-900 truncate max-w-[70%]" title={h.filename}>{h.filename}</div>
                              <span className={`text-xs px-2 py-0.5 rounded-full ${h.is_deepfake ? 'bg-red-50 text-red-700 border border-red-200' : 'bg-green-50 text-green-700 border border-green-200'}`}>
                                {h.is_deepfake ? 'Deepfake' : 'Authentic'}
                              </span>
                            </div>
                            <div className="mt-1 flex items-center justify-between text-xs text-gray-600">
                              <span className="capitalize">{h.file_type}</span>
                              <span>{Math.round(h.confidence * 100)}%</span>
                            </div>
                            <div className="mt-1 text-[10px] text-gray-500">{new Date(h.timestamp).toLocaleString()}</div>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                </div>
              </aside>
            )} */}
          </div>
        </div>
      </div>
    </div>
  )
}

