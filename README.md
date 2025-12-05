# Deepfake Detection Platform

A comprehensive AI-powered deepfake detection platform with user authentication, MongoDB storage, and real-time analysis capabilities.

## Features

- 🔐 **User Authentication**: Secure login/register system with JWT tokens
- 🖼️ **Image Detection**: AI-powered deepfake detection for images
- 🎥 **Video Detection**: Advanced video analysis with temporal consistency checks
- 🎵 **Audio Analysis**: Voice similarity comparison and deepfake detection
- 📊 **History Tracking**: Complete detection history stored in MongoDB
- 🎨 **Modern UI**: Beautiful, responsive interface with animations
- 🔒 **Secure Storage**: All data securely stored in MongoDB with user isolation

## Tech Stack

### Backend
- **FastAPI**: High-performance Python web framework
- **MongoDB**: NoSQL database for flexible data storage
- **Motor**: Async MongoDB driver for FastAPI
- **PyJWT**: JWT token authentication
- **Passlib**: Password hashing and verification
- **PyTorch**: Deep learning models for detection

### Frontend
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first CSS framework
- **Framer Motion**: Smooth animations and transitions
- **Lucide React**: Beautiful icons
- **React Hot Toast**: Toast notifications

## Prerequisites

- Python 3.8+
- Node.js 18+
- MongoDB (local or cloud instance)
- Git



## ⚙️ Requirements

Before installing, ensure you have:

### ✅ Python 3.12  
Download: https://www.python.org/downloads/

### ✅ Node.js 18+  
Download: https://nodejs.org/

### ✅ wkhtmltopdf  
Download: https://wkhtmltopdf.org/downloads.html

After installing wkhtmltopdf, add this to PATH:
```bash
C:\Program Files\wkhtmltopdf\bin
```

Verify installation:

```bash
wkhtmltopdf --version
```


# 🖥️ Automated Setup & Start Scripts

This project includes two Windows batch files for easy installation and launch.

---

## ⚡ **`setup.bat` — One-Time Installation**

Running `setup.bat` will:

- Check if **wkhtmltopdf** is installed (required for PDF generation)  
- Create Python virtual environment automatically  
- Install backend Python dependencies  
- Install frontend Node.js dependencies  
- Prepare the project for first use  

### Usage:

```bash
./setup.bat
```
▶️ start.bat — Start Backend + Frontend
After installation, run the entire platform using:

```bash
./start.bat
```
This will:

-Launch backend → http://localhost:8000
-Launch frontend on port 5555 → http://localhost:5555
-Open two separate terminal windows automatically


## Manual Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd DeepfakeDetection/app
```

### 2. Backend Setup

```bash
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Set up MongoDB database
python setup.py

# Optional: Set environment variables
export MONGODB_URL="mongodb://localhost:27017"
export SECRET_KEY="your-secret-key-here"

# Start the backend server
python main.py
```

The backend will be available at `http://localhost:8000`

### 3. Frontend Setup

```bash
cd frontend

# Install Node.js dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Environment Variables

Create a `.env` file in the backend directory:

```env
# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017

# JWT Configuration
SECRET_KEY=your-super-secret-key-change-this-in-production

# Optional: Production settings
DEBUG=false
HOST=0.0.0.0
PORT=8000
```

## Database Setup

The application uses MongoDB with the following collections:

- **users**: User accounts and authentication data
- **detection_history**: Media detection results
- **audio_references**: Audio comparison results

Run the setup script to initialize the database:

```bash
cd backend
python setup.py
```





## WKHTMLTOPDF Installation (Required for PDF Generation)

The platform uses wkhtmltopdf to generate high-quality PDFs from HTML.
Please install it based on your operating system:

Windows Installation

Download the installer from the official site:
https://wkhtmltopdf.org/downloads.html

Select the Windows .msi file
(e.g., wkhtmltox-0.12.6-1.msvc2015-win64.exe)

Run the installer → Complete setup

Add the installation folder to your PATH:

Example path:

C:\Program Files\wkhtmltopdf\bin


Verify installation:

wkhtmltopdf --version

Linux Installation (Ubuntu/Debian)
Install dependencies:
sudo apt update
sudo apt install -y fontconfig libjpeg-turbo8 xfonts-base xfonts-75dpi

Download wkhtmltopdf:
wget https://github.com/wkhtmltopdf/wkhtmltopdf/releases/download/0.12.6-1/wkhtmltox_0.12.6-1.focal_amd64.deb

Install:
sudo apt install ./wkhtmltox_0.12.6-1.focal_amd64.deb

Verify:
wkhtmltopdf --version

MacOS Installation

The easiest method is Homebrew:

brew install wkhtmltopdf


Or download DMG from:

https://wkhtmltopdf.org/downloads.html

Then verify:

wkhtmltopdf --version

## API Endpoints

### Authentication
- `POST /auth/register` - User registration
- `POST /auth/login` - User login
- `GET /auth/me` - Get current user info

### Detection
- `POST /detect/image` - Image deepfake detection
- `POST /detect/video` - Video deepfake detection
- `POST /detect/video-audio` - Video with audio analysis
- `POST /detect/audio/reference` - Audio similarity comparison

### History
- `GET /history` - Get user's detection history

## Usage

### 1. Registration/Login
- Visit `http://localhost:3000/auth`
- Create a new account or sign in with existing credentials

### 2. Media Detection
- Navigate to the scan page
- Upload images or videos for analysis
- View real-time detection results with confidence scores

### 3. Audio Analysis
- Use the audio tab for voice similarity comparison
- Upload reference and test audio files
- Get similarity scores and match verdicts

### 4. History
- View your complete detection history
- Filter by detection type and date
- Track your analysis patterns

## Project Structure

```
app/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── database.py          # MongoDB models and connection
│   ├── auth.py              # Authentication logic
│   ├── setup.py             # Database setup script
│   ├── requirements.txt     # Python dependencies
│   ├── models/              # AI models
│   ├── checkpoints/         # Model weights
│   └── uploads/             # File uploads
├── frontend/
│   ├── src/
│   │   ├── app/             # Next.js app router
│   │   ├── components/      # React components
│   │   └── styles/          # CSS styles
│   ├── package.json         # Node.js dependencies
│   └── next.config.ts       # Next.js configuration
└── README.md
```

## Security Features

- **JWT Authentication**: Secure token-based authentication
- **Password Hashing**: Bcrypt password encryption
- **User Isolation**: Data is isolated per user
- **Input Validation**: Comprehensive input sanitization
- **CORS Protection**: Cross-origin request protection

## Performance Optimizations

- **Async Operations**: Non-blocking database operations
- **Indexed Queries**: Optimized MongoDB indexes
- **File Streaming**: Efficient file upload handling
- **Caching**: Client-side caching for better UX

## Troubleshooting

### MongoDB Connection Issues
1. Ensure MongoDB is running: `mongod`
2. Check connection string in environment variables
3. Verify network access and firewall settings

### Authentication Issues
1. Check JWT secret key configuration
2. Verify token expiration settings
3. Clear browser local storage if needed

### File Upload Issues
1. Check file size limits (50MB max)
2. Verify supported file formats
3. Ensure proper file permissions

### Model Loading Issues
1. Verify model checkpoint files exist
2. Check PyTorch installation
3. Ensure sufficient GPU memory (if using CUDA)

## Development

### Backend Development
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend Development
```bash
cd frontend
npm install
npm run dev
```

### Database Development
```bash
cd backend
python setup.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation at `http://localhost:8000/docs`

## Acknowledgments

- FastAPI for the excellent web framework
- MongoDB for the flexible database solution
- Next.js for the powerful React framework
- The open-source community for various dependencies
#   C c i t r _ d e e p f a k e D t e c t i o n 
 
 #   d e e p f a k e - d e t e c t i o n - c c i t r 
 
 