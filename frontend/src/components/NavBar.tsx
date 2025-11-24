'use client'

import React, { useState } from 'react'
import Link from 'next/link'
import { useAuth } from './AuthContext'
import { motion } from 'framer-motion'
import { Menu, X, User, LogOut, History } from 'lucide-react'

export default function NavBar() {
  const { user, logout } = useAuth()
  const [isMenuOpen, setIsMenuOpen] = useState(false)

  const handleLogout = () => {
    logout()
    setIsMenuOpen(false)
  }

  return (
    <nav className="bg-gray-300 shadow-lg border-b border-gray-200 fixed top-0 left-0 right-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center ">
          {/* Logo */}
          <Link href="/" className="flex items-center justify-center space-x-2">
          <img src="/LOGO(3).png" alt="Logo" className="h-full w-80" />
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            <Link 
              href="/scan" 
              className="text-gray-700 hover:text-red-600 transition-colors font-medium"
            >
              Scan
            </Link>
            
            {user && (
              <Link 
                href="/history" 
                className="text-gray-700 hover:text-red-600 transition-colors font-medium flex items-center space-x-1"
              >
                <History className="h-4 w-4" />
                <span>History</span>
              </Link>
            )}
             {user ? (
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2 text-gray-700">
                  <User className="h-4 w-4" />
                  <span className="font-medium">{user.username}</span>
                </div>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleLogout}
                  className="flex items-center space-x-1 px-4 py-2 text-red-600 hover:text-red-700 font-medium transition-colors"
                >
                  <LogOut className="h-4 w-4" />
                  <span>Logout</span>
                </motion.button>
              </div>
            ) : (
              <Link 
                href="/auth" 
                className="bg-red-600 text-white px-4  rounded-lg hover:bg-red-700 transition-colors font-medium"
              >
                Sign In
              </Link>
            )}
          </div>

          

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="text-gray-700 hover:text-red-600 transition-colors"
            >
              {isMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden border-t border-gray-200 py-4"
          >
            <div className="space-y-4">
              <Link 
                href="/scan" 
                className="block text-gray-700 hover:text-red-600 transition-colors font-medium"
                onClick={() => setIsMenuOpen(false)}
              >
                Scan
              </Link>
              
              {user && (
                <Link 
                  href="/history" 
                  className="text-gray-700 hover:text-red-600 transition-colors font-medium flex items-center space-x-2"
                  onClick={() => setIsMenuOpen(false)}
                >
                  <History className="h-4 w-4" />
                  <span>History</span>
                </Link>
              )}

              {user ? (
                <div className="space-y-2">
                  <div className="flex items-center space-x-2 text-gray-700 py-2">
                    <User className="h-4 w-4" />
                    <span className="font-medium">{user.username}</span>
                  </div>
                  <button
                    onClick={handleLogout}
                    className="flex items-center space-x-2 text-red-600 hover:text-red-700 font-medium transition-colors w-full"
                  >
                    <LogOut className="h-4 w-4" />
                    <span>Logout</span>
                  </button>
                </div>
              ) : (
                <Link 
                  href="/auth" 
                  className="block bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-colors font-medium text-center"
                  onClick={() => setIsMenuOpen(false)}
                >
                  Sign In
                </Link>
              )}
            </div>
          </motion.div>
        )}
      </div>
    </nav>
  )
}


