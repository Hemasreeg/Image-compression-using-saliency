// AI Portrait Mode - Main JavaScript
// Handles file upload, drag & drop, and AJAX form submission

$(document).ready(function() {
    console.log('AI Portrait Mode - JavaScript Loaded');
    
    // === File Upload Elements ===
    const dropZone = $('#dropZone');
    const fileInput = $('#fileInput');
    const uploadForm = $('#uploadForm');
    const previewSection = $('#previewSection');
    const imagePreview = $('#imagePreview');
    const optionsSection = $('#optionsSection');
    const resultsSection = $('#resultsSection');
    const processingOverlay = $('#processingOverlay');
    const blurStrength = $('#blurStrength');
    const blurValue = $('#blurValue');
    
    console.log('Elements found:', {
        dropZone: dropZone.length,
        fileInput: fileInput.length
    });
    
    // === Drag & Drop Functionality ===
    if (dropZone.length) {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone[0].addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        // Highlight drop zone when dragging over it
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone[0].addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone[0].addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            dropZone.addClass('dragover');
        }
        
        function unhighlight() {
            dropZone.removeClass('dragover');
        }
        
        // Handle dropped files
        dropZone[0].addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0) {
                handleFiles(files);
            }
        }
    }
    
    // === File Input Change ===
    fileInput.on('change', function(e) {
        const files = this.files;
        if (files.length > 0) {
            handleFiles(files);
        }
    });
    
    // === Handle Files ===
    function handleFiles(files) {
        if (files.length === 0) return;
        
        const file = files[0];
        
        // Validate file type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
        if (!validTypes.includes(file.type)) {
            alert('Invalid file type! Please upload a valid image (JPG, PNG, GIF, BMP).');
            return;
        }
        
        // Validate file size (16MB max)
        const maxSize = 16 * 1024 * 1024; // 16MB in bytes
        if (file.size > maxSize) {
            alert('File too large! Maximum size is 16MB.');
            return;
        }
        
        // Preview image
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.attr('src', e.target.result);
            previewSection.removeClass('d-none');
            optionsSection.removeClass('d-none');
            resultsSection.addClass('d-none');
        };
        reader.readAsDataURL(file);
        
        // Set file to input (for drag & drop)
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput[0].files = dataTransfer.files;
    }
    
    // === Remove Preview ===
    $('#removePreview').on('click', function() {
        fileInput.val('');
        previewSection.addClass('d-none');
        optionsSection.addClass('d-none');
        resultsSection.addClass('d-none');
    });
    
    // === Blur Strength Slider ===
    if (blurStrength.length) {
        blurStrength.on('input', function() {
            blurValue.text(this.value);
        });
    }
    
    // === Form Submission (AJAX) ===
    let isProcessing = false; // Prevent double submissions
    let currentRequest = null; // Track current AJAX request
    
    // Remove any existing handlers to prevent double binding
    uploadForm.off('submit');
    
    uploadForm.on('submit', function(e) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        
        console.log('Form submit triggered, isProcessing:', isProcessing);
        
        // Prevent double submission
        if (isProcessing) {
            console.log('BLOCKED: Already processing, ignoring duplicate submit');
            return false;
        }
        
        // Abort any previous request
        if (currentRequest) {
            console.log('Aborting previous request');
            currentRequest.abort();
        }
        
        // Validate file selection
        if (!fileInput[0].files.length) {
            showNotification('Please select an image first!', 'error');
            return;
        }
        
        // Validate effect selection
        const selectedEffects = $('input[name="output_effects"]:checked').length;
        if (selectedEffects === 0) {
            showNotification('Please select at least one effect!', 'error');
            return;
        }
        
        // Set processing flag and disable button
        isProcessing = true;
        $('#processBtn').prop('disabled', true);
        
        // Show processing overlay
        processingOverlay.css('display', 'flex');
        
        // Create FormData
        const formData = new FormData(this);
        
        // AJAX request
        currentRequest = $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                console.log('Upload success:', response);
                
                // Hide processing overlay
                processingOverlay.hide();
                
                if (response.success) {
                    currentRequest = null;
                    // Store results in sessionStorage with accuracy and upload_id
                    const resultsWithAccuracy = {
                        ...response.results,
                        accuracy: response.accuracy || 0,
                        upload_id: response.upload_id
                    };
                    sessionStorage.setItem('processResults', JSON.stringify(resultsWithAccuracy));
                    
                    // Redirect to results page
                    window.location.href = '/results';
                } else {
                    // Reset flag and re-enable button on error
                    currentRequest = null;
                    isProcessing = false;
                    $('#processBtn').prop('disabled', false);
                    showNotification(response.error || 'Processing failed. Please try again.', 'error');
                }
            },
            error: function(xhr, status, error) {
                console.error('Upload error:', error);
                
                // Reset flag and re-enable button
                currentRequest = null;
                isProcessing = false;
                $('#processBtn').prop('disabled', false);
                processingOverlay.hide();
                
                let errorMsg = 'Upload failed! ';
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    errorMsg += xhr.responseJSON.error;
                } else {
                    errorMsg += 'Please try again.';
                }
                showNotification(errorMsg, 'error');
            }
        });
    });
    
    // === Smooth Scroll ===
    $('a[href*="#"]').on('click', function(e) {
        const target = $(this.getAttribute('href'));
        if (target.length) {
            e.preventDefault();
            $('html, body').stop().animate({
                scrollTop: target.offset().top - 80
            }, 1000);
        }
    });
    
    // === Auto-hide alerts ===
    setTimeout(function() {
        $('.alert').fadeOut('slow');
    }, 5000);
    
    // === Image zoom on click (for profile gallery) ===
    $('.upload-card img').on('click', function() {
        const src = $(this).attr('src');
        const modal = $('<div class="modal fade" tabindex="-1">')
            .html(`
                <div class="modal-dialog modal-lg modal-dialog-centered">
                    <div class="modal-content">
                        <div class="modal-header">
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body text-center">
                            <img src="${src}" class="img-fluid" alt="Full size">
                        </div>
                    </div>
                </div>
            `);
        
        $('body').append(modal);
        modal.modal('show');
        modal.on('hidden.bs.modal', function() {
            modal.remove();
        });
    });
});

// === Utility Functions ===

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Show notification
function showNotification(message, type = 'info') {
    const alertClass = type === 'error' ? 'danger' : type;
    const icon = type === 'error' ? 'exclamation-circle' : 'check-circle';
    
    const alert = $(`
        <div class="alert alert-${alertClass} alert-dismissible fade show position-fixed" 
             style="top: 80px; right: 20px; z-index: 9999; min-width: 300px;" 
             role="alert">
            <i class="fas fa-${icon} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `);
    
    $('body').append(alert);
    
    setTimeout(function() {
        alert.fadeOut('slow', function() {
            $(this).remove();
        });
    }, 5000);
}

// Validate image file
function validateImageFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
    const maxSize = 16 * 1024 * 1024; // 16MB
    
    if (!validTypes.includes(file.type)) {
        return {
            valid: false,
            error: 'Invalid file type. Please upload JPG, PNG, GIF, or BMP.'
        };
    }
    
    if (file.size > maxSize) {
        return {
            valid: false,
            error: 'File too large. Maximum size is 16MB.'
        };
    }
    
    return { valid: true };
}

console.log('✓ Main.js loaded successfully');
