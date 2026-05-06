"""
Flask Web Application for AI-Powered Portrait Mode
with Salient Object Detection
"""

import os
from datetime import datetime
import pytz
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename
import torch
import cv2
import numpy as np
from PIL import Image

from model import get_model
from portrait_effect import PortraitEffect
from smart_compression import SmartCompressor, format_file_size
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Initialize Flask app
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# Configuration from environment variables (for Render deployment)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY', '')

# Database configuration - use PostgreSQL on Render, SQLite locally
database_url = os.getenv('DATABASE_URL')
if database_url:
    # Fix PostgreSQL URL format for SQLAlchemy
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///portrait_app.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
app.config['RESULT_FOLDER'] = 'app/static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)


# ============================================================================
# Database Models
# ============================================================================

class User(UserMixin, db.Model):
    """User model for authentication"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    uploads = db.relationship('Upload', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"


class Upload(db.Model):
    """Upload history model"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    original_path = db.Column(db.String(300), nullable=False)
    mask_path = db.Column(db.String(300))
    portrait_path = db.Column(db.String(300))
    duts_path = db.Column(db.String(300))
    blur_only_path = db.Column(db.String(300))
    darken_only_path = db.Column(db.String(300))
    effect_type = db.Column(db.String(50))
    accuracy_score = db.Column(db.Float, default=0.0)
    upload_date = db.Column(db.DateTime, default=lambda: datetime.now(pytz.timezone('Asia/Kolkata')))
    
    def __repr__(self):
        return f"Upload('{self.filename}', '{self.upload_date}')"


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ============================================================================
# AI Model Loader
# ============================================================================

class AIPortraitProcessor:
    """AI-powered portrait processor with U2-Net - Supports ensemble of multiple models"""
    
    def __init__(self, model_paths=None, model_type='u2net', use_ensemble=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = []  # List of models for ensemble
        self.model_paths = model_paths if model_paths else ['models/best_model.pth']
        self.model_type = model_type
        self.use_ensemble = use_ensemble
        
        # Load all available models
        self.load_models()
        
        # Define transform (updated to 416x416 for 90%+ accuracy models)
        self.transform = A.Compose([
            A.Resize(416, 416),  # Match training resolution
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def load_models(self):
        """Load all available trained models for ensemble prediction"""
        print(f"\n{'='*60}")
        print(f"🔄 Loading models for {'ENSEMBLE' if self.use_ensemble else 'SINGLE'} prediction")
        print(f"{'='*60}")
        
        loaded_count = 0
        for model_path in self.model_paths:
            if not os.path.exists(model_path):
                print(f"⚠️  Skipping {model_path} (not found)")
                continue
            
            try:
                print(f"\n📦 Loading: {model_path}")
                
                # Detect if model has novel components
                checkpoint = torch.load(model_path, map_location=self.device)
                state_dict_keys = checkpoint['model_state_dict'].keys()
                has_novel_components = any('bottleneck_ca' in key or 'edge_refine' in key or 'adaptive_fusion' in key 
                                            for key in state_dict_keys)
                
                # Create model with appropriate architecture
                from config import Config
                model = get_model(self.model_type, self.device, use_novel_components=has_novel_components, dropout_rate=Config.DROPOUT_RATE)
                
                # Load weights (strict=False to handle dropout parameter mismatches)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                model.eval()
                
                self.models.append({
                    'model': model,
                    'path': model_path,
                    'has_novel': has_novel_components
                })
                
                model_type_str = "ENHANCED" if has_novel_components else "BASE"
                print(f"   ✓ {model_type_str} model loaded successfully")
                loaded_count += 1
                
            except Exception as e:
                print(f"   ✗ Failed to load {model_path}: {e}")
        
        if loaded_count == 0:
            print(f"\n{'='*60}")
            print("⚠️  WARNING: No models loaded!")
            print("   The app will start, but AI features will not work until models are available.")
            print("   Please train a model or download pre-trained models.")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"✅ Loaded {loaded_count} model(s) successfully")
            if self.use_ensemble and loaded_count > 1:
                print(f"🎯 Ensemble mode: Averaging predictions from {loaded_count} models")
            print(f"{'='*60}\n")
    
    def predict_mask(self, image_path):
        """Predict saliency mask using ensemble of models"""
        if len(self.models) == 0:
            raise ValueError("No models loaded. Please ensure model files exist.")
        
        # Read image
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_h, original_w = original_image.shape[:2]
        
        # Transform for model
        transformed = self.transform(image=original_image)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Predict with ensemble
        all_predictions = []
        with torch.no_grad():
            for model_info in self.models:
                outputs = model_info['model'](input_tensor)
                pred_mask = outputs[0]  # Fusion output
                all_predictions.append(pred_mask)
        
        # Ensemble: Average predictions from all models
        if self.use_ensemble and len(all_predictions) > 1:
            pred_mask = torch.stack(all_predictions).mean(dim=0)
        else:
            pred_mask = all_predictions[0]
        
        # Convert to numpy and resize to original size with better quality
        pred_mask = pred_mask.squeeze().cpu().numpy()
        
        # Use LANCZOS4 for high-quality upsampling (reduces blur)
        pred_mask = cv2.resize(pred_mask, (original_w, original_h), interpolation=cv2.INTER_LANCZOS4)
        pred_mask = np.clip(pred_mask, 0, 1)
        
        # Post-processing: Sharpen mask edges for cleaner output
        # Apply bilateral filter to reduce noise while preserving edges
        pred_mask_8bit = (pred_mask * 255).astype(np.uint8)
        pred_mask_8bit = cv2.bilateralFilter(pred_mask_8bit, d=5, sigmaColor=50, sigmaSpace=50)
        
        # Enhance contrast to sharpen boundaries
        pred_mask_8bit = cv2.convertScaleAbs(pred_mask_8bit, alpha=1.2, beta=-25)
        pred_mask = pred_mask_8bit.astype(np.float32) / 255.0
        pred_mask = np.clip(pred_mask, 0, 1)
        
        # Calculate confidence score (average of high-confidence predictions)
        # Higher values in mask = more confident about salient object
        confidence_score = float(np.mean(pred_mask[pred_mask > 0.5]) if np.any(pred_mask > 0.5) else np.mean(pred_mask))
        accuracy_percentage = round(confidence_score * 100, 2)
        
        return original_image, pred_mask, accuracy_percentage
    
    def process_image(self, image_path, output_effects=['mask', 'portrait', 'duts'], 
                     effect_type='blur_darken', blur_strength=25):
        """
        Process image with multiple output options
        
        Args:
            image_path: Path to input image
            output_effects: List of effects to generate ['mask', 'portrait', 'duts', 'blur_only', 'darken_only']
            effect_type: Type of portrait effect
            blur_strength: Blur intensity
        
        Returns:
            Dictionary with paths to generated images
        """
        # Predict mask
        original_image, saliency_mask, accuracy = self.predict_mask(image_path)
        
        results = {
            'original': image_path,
            'accuracy': accuracy
        }
        
        # Generate base filename
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save mask (always generated)
        if 'mask' in output_effects:
            mask_filename = f"{base_filename}_mask_{timestamp}.png"
            mask_path = os.path.join(app.config['RESULT_FOLDER'], mask_filename)
            mask_image = (saliency_mask * 255).astype(np.uint8)
            cv2.imwrite(mask_path, mask_image)
            results['mask'] = mask_path
        
        # Save DUTS-style output (binary mask with original image)
        if 'duts' in output_effects:
            duts_filename = f"{base_filename}_duts_{timestamp}.png"
            duts_path = os.path.join(app.config['RESULT_FOLDER'], duts_filename)
            # Create DUTS visualization (original image with binary mask overlay)
            binary_mask = (saliency_mask > 0.5).astype(np.uint8) * 255
            duts_vis = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(duts_path, binary_mask)
            results['duts'] = duts_path
        
        # Generate portrait effects
        if 'portrait' in output_effects:
            portrait_generator = PortraitEffect(
                effect_type=effect_type,
                blur_strength=blur_strength,
                blur_sigma=10,
                background_brightness=0.3
            )
            
            portrait_image = portrait_generator.apply_effect(original_image, saliency_mask)
            
            portrait_filename = f"{base_filename}_portrait_{timestamp}.png"
            portrait_path = os.path.join(app.config['RESULT_FOLDER'], portrait_filename)
            portrait_rgb = cv2.cvtColor(portrait_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(portrait_path, portrait_rgb)
            results['portrait'] = portrait_path
        
        # Additional effects
        if 'blur_only' in output_effects:
            blur_gen = PortraitEffect(effect_type='blur', blur_strength=blur_strength)
            blur_image = blur_gen.apply_effect(original_image, saliency_mask)
            blur_filename = f"{base_filename}_blur_{timestamp}.png"
            blur_path = os.path.join(app.config['RESULT_FOLDER'], blur_filename)
            cv2.imwrite(blur_path, cv2.cvtColor(blur_image, cv2.COLOR_RGB2BGR))
            results['blur_only'] = blur_path
        
        if 'darken_only' in output_effects:
            darken_gen = PortraitEffect(effect_type='darken')
            darken_image = darken_gen.apply_effect(original_image, saliency_mask)
            darken_filename = f"{base_filename}_darken_{timestamp}.png"
            darken_path = os.path.join(app.config['RESULT_FOLDER'], darken_filename)
            cv2.imwrite(darken_path, cv2.cvtColor(darken_image, cv2.COLOR_RGB2BGR))
            results['darken_only'] = darken_path
        
        return results


# Initialize AI processor with ensemble of all available models
import os

# List all available models in priority order
model_paths = [
    'models/enhanced_u2net_with_allupdated.pth',    # New: All updates, good result
    'models/enhanced_u2net_with_pre_trained.pth',   # Best: 86.06% F1
    'models/enhanced_u2net.pth',                    # Enhanced model
    'models/best_model.pth'                         # Base pretrained U2-Net
]

# Filter to only existing models
available_models = [path for path in model_paths if os.path.exists(path)]

print(f"\n🤖 Initializing AI Portrait Processor...")
try:
    if not available_models:
        # Create processor without models - it will warn but not crash
        ai_processor = AIPortraitProcessor(model_paths=['models/best_model.pth'], use_ensemble=True)
    else:
        ai_processor = AIPortraitProcessor(model_paths=available_models, use_ensemble=True)
except Exception as e:
    print(f"⚠️  Error initializing processor: {e}")
    print("   App will continue to run. Models can be uploaded/trained later.")
    # Create a dummy processor with no models
    class DummyProcessor:
        models = []
    ai_processor = DummyProcessor()


# ============================================================================
# Helper Functions
# ============================================================================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def models_available():
    """Check if AI models are loaded and available"""
    return hasattr(ai_processor, 'models') and len(ai_processor.models) > 0


# ============================================================================
# Routes
# ============================================================================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Validation
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
        
        # Create user
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, email=email, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user, remember=request.form.get('remember'))
            flash(f'Welcome back, {user.username}!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login failed. Please check email and password.', 'danger')
    
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


@app.route('/home')
@login_required
def home():
    """User home page with project guide"""
    return render_template('home.html')


@app.route('/results')
@login_required
def results():
    """Results display page"""
    return render_template('results.html')


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """Image upload and processing page"""
    if request.method == 'POST':
        print(f"\n{'='*60}")
        print(f"UPLOAD REQUEST RECEIVED at {datetime.now()}")
        print(f"{'='*60}")
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{current_user.id}_{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Get processing options
            output_effects = request.form.getlist('output_effects')
            effect_type = request.form.get('effect_type', 'blur_darken')
            blur_strength = int(request.form.get('blur_strength', 25))
            
            # Process image
            try:
                # Check for recent duplicate upload (within 5 seconds)
                from datetime import timedelta
                recent_threshold = datetime.now(pytz.timezone('Asia/Kolkata')) - timedelta(seconds=5)
                existing_upload = Upload.query.filter_by(
                    user_id=current_user.id,
                    filename=filename
                ).filter(Upload.upload_date >= recent_threshold).first()
                
                if existing_upload:
                    # Duplicate detected, return existing record
                    print(f"Duplicate upload detected for {filename}, returning existing record")
                    return jsonify({
                        'success': True,
                        'upload_id': existing_upload.id,
                        'accuracy': existing_upload.accuracy_score or 0.0,
                        'results': {
                            'original': url_for('static', filename=f'uploads/{os.path.basename(existing_upload.original_path)}'),
                            'mask': url_for('static', filename=f'results/{os.path.basename(existing_upload.mask_path)}') if existing_upload.mask_path else None,
                            'portrait': url_for('static', filename=f'results/{os.path.basename(existing_upload.portrait_path)}') if existing_upload.portrait_path else None,
                            'duts': url_for('static', filename=f'results/{os.path.basename(existing_upload.duts_path)}') if existing_upload.duts_path else None,
                            'blur_only': url_for('static', filename=f'results/{os.path.basename(existing_upload.blur_only_path)}') if existing_upload.blur_only_path else None,
                            'darken_only': url_for('static', filename=f'results/{os.path.basename(existing_upload.darken_only_path)}') if existing_upload.darken_only_path else None,
                        }
                    })
                
                # Check if models are available
                if not models_available():
                    flash('⚠️ AI models are not loaded. Please upload model files or train a model.', 'warning')
                    return redirect(url_for('upload'))
                
                results = ai_processor.process_image(
                    filepath,
                    output_effects=output_effects if output_effects else ['mask', 'portrait', 'duts'],
                    effect_type=effect_type,
                    blur_strength=blur_strength
                )
                
                # Save to database
                upload_record = Upload(
                    user_id=current_user.id,
                    filename=filename,
                    original_path=filepath,
                    mask_path=results.get('mask'),
                    portrait_path=results.get('portrait'),
                    duts_path=results.get('duts'),
                    blur_only_path=results.get('blur_only'),
                    darken_only_path=results.get('darken_only'),
                    effect_type=effect_type,
                    accuracy_score=results.get('accuracy', 0.0)
                )
                db.session.add(upload_record)
                db.session.commit()
                print(f"New upload saved: ID={upload_record.id}, File={filename}")
                
                # Return results
                return jsonify({
                    'success': True,
                    'upload_id': upload_record.id,
                    'accuracy': results.get('accuracy', 0.0),
                    'results': {
                        'original': url_for('static', filename=f'uploads/{unique_filename}'),
                        'mask': url_for('static', filename=f'results/{os.path.basename(results["mask"])}') if 'mask' in results else None,
                        'portrait': url_for('static', filename=f'results/{os.path.basename(results["portrait"])}') if 'portrait' in results else None,
                        'duts': url_for('static', filename=f'results/{os.path.basename(results["duts"])}') if 'duts' in results else None,
                        'blur_only': url_for('static', filename=f'results/{os.path.basename(results["blur_only"])}') if 'blur_only' in results else None,
                        'darken_only': url_for('static', filename=f'results/{os.path.basename(results["darken_only"])}') if 'darken_only' in results else None,
                    }
                })
            
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        return jsonify({'error': 'Invalid file type'}), 400
    
    return render_template('upload.html')


@app.route('/profile')
@login_required
def profile():
    """User profile with upload history"""
    uploads = Upload.query.filter_by(user_id=current_user.id).order_by(Upload.upload_date.desc()).all()
    
    # Convert file system paths to URL paths
    for upload in uploads:
        # Convert original image path
        if upload.original_path:
            upload.original_path = url_for('static', filename='uploads/' + os.path.basename(upload.original_path))
        # Convert paths to relative URLs
        if upload.portrait_path:
            upload.portrait_path = url_for('static', filename='results/' + os.path.basename(upload.portrait_path))
        if upload.mask_path:
            upload.mask_path = url_for('static', filename='results/' + os.path.basename(upload.mask_path))
        if upload.duts_path:
            upload.duts_path = url_for('static', filename='results/' + os.path.basename(upload.duts_path))
        if upload.blur_only_path:
            upload.blur_only_path = url_for('static', filename='results/' + os.path.basename(upload.blur_only_path))
        if upload.darken_only_path:
            upload.darken_only_path = url_for('static', filename='results/' + os.path.basename(upload.darken_only_path))
    
    return render_template('profile.html', uploads=uploads)


@app.route('/update_password', methods=['POST'])
@login_required
def update_password():
    """Update user password"""
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    
    if bcrypt.check_password_hash(current_user.password, current_password):
        hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
        current_user.password = hashed_password
        db.session.commit()
        flash('Password updated successfully!', 'success')
    else:
        flash('Current password is incorrect', 'danger')
    
    return redirect(url_for('profile'))


@app.route('/delete_upload/<int:upload_id>', methods=['POST'])
@login_required
def delete_upload(upload_id):
    """Delete an upload from history"""
    upload = Upload.query.get_or_404(upload_id)
    
    # Check ownership
    if upload.user_id != current_user.id:
        flash('Unauthorized action', 'danger')
        return redirect(url_for('profile'))
    
    # Delete files
    for path in [upload.original_path, upload.mask_path, upload.portrait_path, upload.duts_path]:
        if path and os.path.exists(path):
            os.remove(path)
    
    # Delete database record
    db.session.delete(upload)
    db.session.commit()
    
    flash('Upload deleted successfully', 'success')
    return redirect(url_for('profile'))


@app.route('/compress/<int:upload_id>/<level>')
@login_required
def compress_image(upload_id, level):
    """
    Compress uploaded image with smart compression
    
    Args:
        upload_id: ID of the upload
        level: Compression level ('low', 'medium', 'high')
    """
    # Validate compression level
    if level not in ['low', 'medium', 'high']:
        return jsonify({'error': 'Invalid compression level'}), 400
    
    # Get upload record
    upload = Upload.query.get_or_404(upload_id)
    
    # Check ownership
    if upload.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        # Check if models are available
        if not models_available():
            return jsonify({'error': 'AI models not loaded. Please upload model files.'}), 503
        
        # Get paths
        original_path = upload.original_path
        mask_path = upload.mask_path
        
        # If mask doesn't exist, generate it
        if not mask_path or not os.path.exists(mask_path):
            _, mask_array, _ = ai_processor.predict_mask(original_path)
            # Save mask temporarily
            mask_filename = f"temp_mask_{upload_id}.png"
            mask_path = os.path.join(app.config['RESULT_FOLDER'], mask_filename)
            cv2.imwrite(mask_path, (mask_array * 255).astype(np.uint8))
        
        # Create compressor
        compressor = SmartCompressor(level=level)
        
        # Generate output filename
        base_filename = os.path.splitext(os.path.basename(upload.filename))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        compressed_filename = f"{base_filename}_compressed_{level}_{timestamp}.jpg"
        compressed_path = os.path.join(app.config['RESULT_FOLDER'], compressed_filename)
        
        # Compress and save without bounding box
        stats = compressor.compress_and_save(original_path, mask_path, compressed_path, draw_box=False)
        
        # Return statistics and download URL
        return jsonify({
            'success': True,
            'filename': compressed_filename,
            'download_url': url_for('static', filename=f'results/{compressed_filename}'),
            'stats': {
                'original_size': format_file_size(stats['original_size']),
                'compressed_size': format_file_size(stats['compressed_size']),
                'reduction': format_file_size(stats['reduction']),
                'reduction_percent': f"{stats['reduction_percent']:.1f}%",
                'level': level,
                'description': stats['description']
            }
        })
        
    except Exception as e:
        print(f"Compression error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Initialize Database
# ============================================================================

def init_db():
    """Initialize database"""
    with app.app_context():
        db.create_all()
        print("✓ Database initialized")


# ============================================================================
# Run Application
# ============================================================================

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
