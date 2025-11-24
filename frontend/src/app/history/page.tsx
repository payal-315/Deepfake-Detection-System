'use client'

import React, { useState, useEffect } from 'react'
import { useAuth } from '../../components/AuthContext'
import { useRouter } from 'next/navigation'
import NavBar from '../../components/NavBar'
import { motion } from 'framer-motion'
import { History, FileText, Video, Image, Calendar, Clock, TrendingUp, Shield, AlertTriangle } from 'lucide-react'

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

interface AudioReferenceItem {
  _id: string
  reference_filename: string
  test_filename: string
  similarity: number
  verdict: string
  timestamp: string
}

export default function HistoryPage() {
  const { user, token, loading } = useAuth()
  const router = useRouter()
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [audioReferences, setAudioReferences] = useState<AudioReferenceItem[]>([])
  const [loadingHistory, setLoadingHistory] = useState(true)
  const [activeTab, setActiveTab] = useState<'detections' | 'audio'>('detections')

  useEffect(() => {
    if (!loading && !user) {
      router.push('/auth')
    }
  }, [user, loading, router])

  useEffect(() => {
    if (user && token) {
      loadHistory()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user, token])

  const loadHistory = async () => {
    if (!token) return
    
    try {
      setLoadingHistory(true)
      const res = await fetch('http://localhost:8000/history', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
      
      if (res.ok) {
        const data = await res.json()
        setHistory(data.detections || [])
        setAudioReferences(data.audio_references || [])
      }
    } catch (error) {
      console.error('Error loading history:', error)
    } finally {
      setLoadingHistory(false)
    }
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

  const getFileIcon = (fileType: string) => {
    return fileType === 'video' ? <Video className="h-5 w-5" /> : <Image className="h-5 w-5" />
  }

  const getStatusIcon = (isDeepfake: boolean) => {
    return isDeepfake ? 
      <AlertTriangle className="h-5 w-5 text-red-500" /> : 
      <Shield className="h-5 w-5 text-green-500" />
  }

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <NavBar />
      
      <div className="pt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8"
          >
            <div className="flex items-center space-x-3 mb-4">
              <History className="h-8 w-8 text-red-600" />
              <h1 className="text-3xl font-bold text-gray-900">Detection History</h1>
            </div>
            <p className="text-gray-600">View your past deepfake detection results and audio comparisons.</p>
          </motion.div>

          {/* Tabs */}
          <div className="mb-6">
            <div className="inline-flex items-center gap-2 rounded-xl border border-gray-200 p-1 bg-white">
              <button
                onClick={() => setActiveTab('detections')}
                className={`${activeTab === 'detections' ? 'bg-red-600 text-white' : 'bg-white text-gray-700 hover:bg-gray-50'} px-4 py-2 rounded-lg text-sm font-medium`}
              >
                Media Detections ({history.length})
              </button>
              <button
                onClick={() => setActiveTab('audio')}
                className={`${activeTab === 'audio' ? 'bg-red-600 text-white' : 'bg-white text-gray-700 hover:bg-gray-50'} px-4 py-2 rounded-lg text-sm font-medium`}
              >
                Audio References ({audioReferences.length})
              </button>
            </div>
          </div>

          {/* Content */}
          {loadingHistory ? (
            <div className="flex items-center justify-center py-12">
              <div className="text-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-red-600 mx-auto mb-4"></div>
                <p className="text-gray-600">Loading history...</p>
              </div>
            </div>
          ) : (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              {activeTab === 'detections' && (
                <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden">
                  {history.length === 0 ? (
                    <div className="text-center py-12">
                      <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                      <h3 className="text-lg font-medium text-gray-900 mb-2">No detections yet</h3>
                      <p className="text-gray-600">Start by uploading an image or video for analysis.</p>
                    </div>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead className="bg-gray-50 border-b border-gray-200">
                          <tr>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">File</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Result</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Processing Time</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                          </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                          {history.map((item, index) => (
                            <motion.tr
                              key={item._id}
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ delay: index * 0.05 }}
                              className="hover:bg-gray-50"
                            >
                              <td className="px-6 py-4 whitespace-nowrap">
                                <div className="flex items-center">
                                  <div className="flex-shrink-0 h-10 w-10">
                                    <div className="h-10 w-10 rounded-lg bg-gray-100 flex items-center justify-center">
                                      {getFileIcon(item.file_type)}
                                    </div>
                                  </div>
                                  <div className="ml-4">
                                    <div className="text-sm font-medium text-gray-900 truncate max-w-xs">
                                      {item.filename}
                                    </div>
                                  </div>
                                </div>
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap">
                                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800 capitalize">
                                  {item.file_type}
                                </span>
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap">
                                <div className="flex items-center">
                                  {getStatusIcon(item.is_deepfake)}
                                  <span className={`ml-2 text-sm font-medium ${item.is_deepfake ? 'text-red-700' : 'text-green-700'}`}>
                                    {item.is_deepfake ? 'Deepfake' : 'Authentic'}
                                  </span>
                                </div>
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap">
                                <div className="flex items-center">
                                  <TrendingUp className="h-4 w-4 text-gray-400 mr-1" />
                                  <span className="text-sm text-gray-900">
                                    {Math.round(item.confidence * 100)}%
                                  </span>
                                </div>
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap">
                                <div className="flex items-center">
                                  <Clock className="h-4 w-4 text-gray-400 mr-1" />
                                  <span className="text-sm text-gray-900">
                                    {item.processing_time.toFixed(2)}s
                                  </span>
                                </div>
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap">
                                <div className="flex items-center">
                                  <Calendar className="h-4 w-4 text-gray-400 mr-1" />
                                  <span className="text-sm text-gray-900">
                                    {formatDate(item.timestamp)}
                                  </span>
                                </div>
                              </td>
                            </motion.tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}

              {activeTab === 'audio' && (
                <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden">
                  {audioReferences.length === 0 ? (
                    <div className="text-center py-12">
                      <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                      <h3 className="text-lg font-medium text-gray-900 mb-2">No audio references yet</h3>
                      <p className="text-gray-600">Start by comparing audio files for similarity analysis.</p>
                    </div>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead className="bg-gray-50 border-b border-gray-200">
                          <tr>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Reference</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Test</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Similarity</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Verdict</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                          </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                          {audioReferences.map((item, index) => (
                            <motion.tr
                              key={item._id}
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ delay: index * 0.05 }}
                              className="hover:bg-gray-50"
                            >
                              <td className="px-6 py-4 whitespace-nowrap">
                                <div className="text-sm font-medium text-gray-900 truncate max-w-xs">
                                  {item.reference_filename}
                                </div>
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap">
                                <div className="text-sm font-medium text-gray-900 truncate max-w-xs">
                                  {item.test_filename}
                                </div>
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap">
                                <div className="flex items-center">
                                  <TrendingUp className="h-4 w-4 text-gray-400 mr-1" />
                                  <span className="text-sm text-gray-900">
                                    {Math.round(item.similarity * 100)}%
                                  </span>
                                </div>
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap">
                                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                  item.verdict === 'match' 
                                    ? 'bg-green-100 text-green-800' 
                                    : 'bg-red-100 text-red-800'
                                }`}>
                                  {item.verdict}
                                </span>
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap">
                                <div className="flex items-center">
                                  <Calendar className="h-4 w-4 text-gray-400 mr-1" />
                                  <span className="text-sm text-gray-900">
                                    {formatDate(item.timestamp)}
                                  </span>
                                </div>
                              </td>
                            </motion.tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}
            </motion.div>
          )}
        </div>
      </div>
    </div>
  )
}
