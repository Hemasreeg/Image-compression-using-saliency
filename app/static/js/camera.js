// AI Portrait Mode - Camera Capture
// Handles webcam access and photo capture

let videoStream = null;
let cameraActive = false;

// === Camera Modal Events ===
const cameraModal = document.getElementById('cameraModal');
console.log('Camera modal element:', cameraModal);

if (cameraModal) {
    cameraModal.addEventListener('shown.bs.modal', function () {
        console.log('Camera modal opened');
        startCamera();
    });

    cameraModal.addEventListener('hidden.bs.modal', function () {
        console.log('Camera modal closed');
        stopCamera();
    });
} else {
    console.error('Camera modal not found in DOM');
}

// === Start Camera ===
async function startCamera() {
    const video = document.getElementById('cameraVideo');
    
    if (!video) {
        console.error('Video element not found');
        alert('Camera video element not found. Please refresh the page.');
        return;
    }
    
    try {
        console.log('Requesting camera access...');
        console.log('Navigator.mediaDevices:', navigator.mediaDevices);
        
        // Check if getUserMedia is available
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Camera API not available in this browser');
        }
        
        // Request camera access with fallback constraints
        const constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user' // Front camera
            },
            audio: false
        };
        
        console.log('Requesting camera with constraints:', constraints);
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        
        console.log('Camera access granted, stream:', stream);
        
        // Set video source
        video.srcObject = stream;
        videoStream = stream;
        cameraActive = true;
        
        // Play video
        await video.play();
        console.log('Video playing successfully');
        
    } catch (error) {
        console.error('Camera access error:', error);
        console.error('Error name:', error.name);
        console.error('Error message:', error.message);
        
        let errorMessage = 'Unable to access camera. ';
        
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            errorMessage += 'Please allow camera permissions in your browser settings.';
        } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
            errorMessage += 'No camera device found. Please connect a camera and try again.';
        } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
            errorMessage += 'Camera is already in use by another application.';
        } else if (error.name === 'OverconstrainedError' || error.name === 'ConstraintNotSatisfiedError') {
            errorMessage += 'Camera does not support the requested settings.';
        } else if (error.name === 'TypeError') {
            errorMessage += 'Camera API not supported. Please use Chrome, Firefox, or Edge.';
        } else {
            errorMessage += error.message || 'Unknown error occurred.';
        }
        
        alert(errorMessage);
        
        // Close modal on error
        const modalInstance = bootstrap.Modal.getInstance(cameraModal);
        if (modalInstance) {
            modalInstance.hide();
        }
    }
}

// === Stop Camera ===
function stopCamera() {
    if (videoStream) {
        console.log('Stopping camera...');
        
        // Stop all tracks
        videoStream.getTracks().forEach(track => {
            track.stop();
        });
        
        const video = document.getElementById('cameraVideo');
        if (video) {
            video.srcObject = null;
        }
        
        videoStream = null;
        cameraActive = false;
        
        console.log('Camera stopped');
    }
}

// === Capture Photo ===
const captureBtn = document.getElementById('captureBtn');
if (captureBtn) {
    captureBtn.addEventListener('click', function() {
        capturePhoto();
    });
}

function capturePhoto() {
    const video = document.getElementById('cameraVideo');
    const canvas = document.getElementById('cameraCanvas');
    
    if (!video || !canvas) {
        console.error('Video or canvas element not found');
        return;
    }
    
    if (!cameraActive) {
        alert('Camera is not active. Please allow camera access.');
        return;
    }
    
    console.log('Capturing photo...');
    
    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw current video frame to canvas
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert canvas to blob
    canvas.toBlob(function(blob) {
        if (!blob) {
            console.error('Failed to create image blob');
            alert('Failed to capture photo. Please try again.');
            return;
        }
        
        console.log('Photo captured:', blob.size, 'bytes');
        
        // Create File object from blob
        const file = new File([blob], 'camera-capture.jpg', {
            type: 'image/jpeg',
            lastModified: Date.now()
        });
        
        // Set to file input
        const fileInput = document.getElementById('fileInput');
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;
        
        // Preview image
        const reader = new FileReader();
        reader.onload = function(e) {
            const imagePreview = document.getElementById('imagePreview');
            const previewSection = document.getElementById('previewSection');
            const optionsSection = document.getElementById('optionsSection');
            const resultsSection = document.getElementById('resultsSection');
            
            imagePreview.src = e.target.result;
            previewSection.classList.remove('d-none');
            optionsSection.classList.remove('d-none');
            resultsSection.classList.add('d-none');
        };
        reader.readAsDataURL(file);
        
        // Close camera modal
        const modalInstance = bootstrap.Modal.getInstance(cameraModal);
        if (modalInstance) {
            modalInstance.hide();
        }
        
        // Show success notification
        if (typeof showNotification === 'function') {
            showNotification('Photo captured successfully!', 'success');
        }
        
        console.log('Photo set to file input');
        
    }, 'image/jpeg', 0.95); // JPEG quality 95%
}

// === Check Camera Support ===
function checkCameraSupport() {
    // Just check if the API exists, don't disable the button
    // Let the startCamera function handle errors
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.warn('Camera API not available');
        return false;
    }
    
    console.log('Camera API available');
    return true;
}

// === Initialize ===
document.addEventListener('DOMContentLoaded', function() {
    console.log('Camera.js loaded');
    checkCameraSupport();
});

// === Cleanup on page unload ===
window.addEventListener('beforeunload', function() {
    stopCamera();
});

console.log('✓ Camera.js loaded successfully');
