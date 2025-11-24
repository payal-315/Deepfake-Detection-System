'use client'

import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, Camera, Video, X, File, Sparkles } from 'lucide-react'
import toast from 'react-hot-toast'
import axios from 'axios'

interface FileUploadProps {
  onDetectionStart: () => void
  onDetectionComplete: (result: unknown) => void
  isProcessing: boolean
  mode?: 'all' | 'video-only'
  token?: string | null
  facedetect: boolean
}

export default function FileUpload({ onDetectionStart, onDetectionComplete, isProcessing, mode = 'all', token, facedetect = false }: FileUploadProps) {

  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (!file) return

    // Validate file type
    const isImage = file.type.startsWith('image/')
    const isVideo = file.type.startsWith('video/')

    if (mode === 'video-only' && !isVideo) {
      toast.error('Please upload a video file')
      return
    }

    if (mode === 'all' && !isImage && !isVideo) {
      toast.error('Please upload an image or video file')
      return
    }

    // Validate file size (50MB limit)
    if (file.size > 50 * 1024 * 1024) {
      toast.error('File size must be less than 50MB')
      return
    }

    setUploadedFile(file)

    // Create preview
    if (isImage) {
      const reader = new FileReader()
      reader.onload = () => setPreview(reader.result as string)
      reader.readAsDataURL(file)
    } else {
      setPreview(URL.createObjectURL(file))
    }

    toast.success('File uploaded successfully!')
  }, [mode])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: mode === 'video-only' ? {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv']
    } : {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp', '.gif'],
      'video/*': ['.mp4', '.avi', '.mov', '.mkv']
    },
    maxFiles: 1,
    disabled: isProcessing
  })

  const handleAnalyze = async () => {
    if (!uploadedFile) return

    onDetectionStart()

    try {
      const formData = new FormData()
      formData.append('file', uploadedFile)

      const endpoint = mode === 'video-only'
        ? (facedetect ? '/detect/video-audiof': '/detect/video-audio')
        : (uploadedFile.type.startsWith('image/') ? '/detect/image' : '/detect/video')

  
      const headers: Record<string, string> = { 'Content-Type': 'multipart/form-data' }
      if (token) {
        headers['Authorization'] = `Bearer ${token}`
      }

      toast.promise(
        axios.post(`http://localhost:8000${endpoint}`, formData, {
          headers,
          timeout: 12000000,
        }),
        {
          loading: 'Analyzing with AI...',
          success: 'Analysis complete!',
          error: 'Analysis failed',
        }
      ).then((response) => {
        onDetectionComplete(response.data)

      }).catch((error: unknown) => {
        console.error('Upload error:', error)
        toast.error((error as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Failed to process file')
      })

    } catch (error: unknown) {
      console.error('Upload error:', error)
      toast.error('Failed to process file')
    }
  }

  const removeFile = () => {
    setUploadedFile(null)
    setPreview(null)
    if (preview && preview.startsWith('blob:')) {
      URL.revokeObjectURL(preview)
    }
  }

  const getFileIcon = () => {
    if (!uploadedFile) return <Upload className="h-12 w-12" />
    return uploadedFile.type.startsWith('image/') ?
      <Camera className="h-12 w-12" /> :
      <Video className="h-12 w-12" />
  }

  return (
    <div className="space-y-6 mt-10">
      {/* Drop Zone */}
      <motion.div
        {...(getRootProps() as Record<string, unknown>)}
        className={`
          relative border-2 border-dashed rounded-3xl p-12 text-center cursor-pointer transition-all duration-300
          ${isDragActive
            ? 'border-red-400 bg-red-50 scale-105'
            : 'border-gray-300 bg-white hover:bg-gray-50 hover:border-gray-400'
          }
          ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}
        `}
        whileHover={!isProcessing ? { scale: 1.02 } : {}}
        whileTap={!isProcessing ? { scale: 0.98 } : {}}
      >
        <input {...getInputProps()} />

        <AnimatePresence mode="wait">
          {isDragActive ? (
            <motion.div
              key="drag-active"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              className="space-y-6"
            >
              <div className="flex justify-center">
                <div className="p-6 bg-red-500/20 rounded-full">
                  <Upload className="h-12 w-12 text-red-400 animate-bounce" />
                </div>
              </div>
              <div>
                <h3 className="text-2xl font-bold text-gray-900 mb-2">Drop it here!</h3>
                <p className="text-red-700">Release to upload your file</p>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="drag-inactive"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              className="space-y-6"
            >
              <div className="flex justify-center">
                <motion.div
                  className="p-6 bg-gradient-to-r from-red-500/20 to-pink-500/20 rounded-full"
                  whileHover={{ rotate: 360 }}
                  transition={{ duration: 0.5 }}
                >
                  {getFileIcon()}
                  <div className="text-gray-900">
                    <Sparkles className="absolute top-2 right-2 h-4 w-4 text-yellow-400 animate-pulse" />
                  </div>
                </motion.div>
              </div>

              <div>
                <h3 className="text-2xl font-bold text-gray-900 mb-4">
                  {uploadedFile ? 'Upload Another File' : (mode === 'video-only' ? 'Upload Your Video' : 'Upload Your Media')}
                </h3>
                <p className="text-gray-700 mb-4">
                  Drag and drop your image or video here, or click to browse
                </p>
                <div className="flex flex-wrap justify-center gap-2 text-sm text-gray-600">
                  {mode === 'video-only' ? (
                    <>
                      <span className="px-3 py-1 bg-gray-100 rounded-full">MP4</span>
                      <span className="px-3 py-1 bg-gray-100 rounded-full">AVI</span>
                      <span className="px-3 py-1 bg-gray-100 rounded-full">MOV</span>
                    </>
                  ) : (
                    <>
                      <span className="px-3 py-1 bg-gray-100 rounded-full">JPG</span>
                      <span className="px-3 py-1 bg-gray-100 rounded-full">PNG</span>
                      <span className="px-3 py-1 bg-gray-100 rounded-full">MP4</span>
                      <span className="px-3 py-1 bg-gray-100 rounded-full">AVI</span>
                      <span className="px-3 py-1 bg-gray-100 rounded-full">+ more</span>
                    </>
                  )}
                </div>
                <p className="text-xs text-gray-500 mt-2">Max file size: 50MB</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* File Preview */}
      <AnimatePresence>
        {uploadedFile && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -20, scale: 0.9 }}
            className="bg-white rounded-2xl p-6 mt-8 border border-gray-200"
          >
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-4">
                <div className="p-3 bg-gradient-to-r from-red-500 to-pink-500 rounded-xl">
                  {uploadedFile.type.startsWith('image/') ?
                    <Camera className="h-6 w-6 text-white" /> :
                    <Video className="h-6 w-6 text-white" />
                  }
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900">{uploadedFile.name}</h4>
                  <p className="text-gray-600 text-sm">
                    {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB • {uploadedFile.type.split('/')[0].toUpperCase()}
                  </p>
                </div>
              </div>
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={removeFile}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                disabled={isProcessing}
              >
                <X className="h-5 w-5 text-gray-500" />
              </motion.button>
            </div>

            {/* Preview */}
            {preview && (
              <div className="mb-6 flex justify-center">
                {uploadedFile.type.startsWith('image/') ? (
                  <motion.img
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    src={preview}
                    alt="Preview"
                    className=" h-100 object-cover rounded-xl border border-gray-200"
                  />
                ) : (
                  <motion.video
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    src={preview}
                    controls
                    className="h-100 object-cover rounded-xl border border-gray-200"
                  />
                )}
              </div>
            )}

            {/* Analyze Button */}
            <motion.button
              whileHover={!isProcessing ? { scale: 1.02, boxShadow: "0 10px 30px rgba(220, 38, 38, 0.4)" } : {}}
              whileTap={!isProcessing ? { scale: 0.98 } : {}}
              onClick={handleAnalyze}
              disabled={isProcessing}
              className={`
                w-full py-4 px-6 rounded-xl font-semibold text-lg transition-all duration-300
                ${isProcessing
                  ? 'bg-gray-200 text-gray-500 cursor-not-allowed'
                  : 'bg-red-600 text-white hover:bg-red-700 shadow-lg'
                }
              `}
            >
              {isProcessing ? (
                <div className="flex items-center justify-center space-x-3">
                  <div className="w-5 h-5 border-2 border-gray-400/50 border-t-gray-600 rounded-full animate-spin"></div>
                  <span>Analyzing with AI...</span>
                </div>
              ) : (
                <div className="flex items-center justify-center space-x-2">
                  <Sparkles className="h-5 w-5" />
                  <span>Analyze with AI</span>
                </div>
              )}
            </motion.button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
