"""
Golf Swing Analysis Web App
Serves comparison dashboard + live video analysis via SwingNet and CBAM-SwingNet.
"""

import os
import sys
import tempfile
import uuid
import base64

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from flask import Flask, request, jsonify, send_from_directory

# Import root model first, then CBAM model
from model import EventDetector

# Temporarily add cbam_swingnet to path for its internal imports
_cbam_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cbam_swingnet')
sys.path.insert(0, _cbam_dir)
from cbam_model import CBAMEventDetector
sys.path.pop(0)

app = Flask(__name__, static_folder='comparison', static_url_path='/comparison')

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'm4v', 'MP4', 'MOV'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

EVENT_NAMES = {
    0: 'Address',
    1: 'Toe-up',
    2: 'Mid-backswing',
    3: 'Top',
    4: 'Mid-downswing',
    5: 'Impact',
    6: 'Mid-follow-through',
    7: 'Finish'
}

TARGET_FPS = 30

# Global models
swingnet_model = None
cbam_model = None
device = None


class VideoDataset(Dataset):
    """Video dataset with optional FPS normalization."""
    def __init__(self, path, input_size=160, transform=None, target_fps=TARGET_FPS):
        self.path = path
        self.input_size = input_size
        self.transform = transform
        self.target_fps = target_fps

        cap = cv2.VideoCapture(path)
        self.original_fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if self.original_fps > 45:
            self.frame_skip = round(self.original_fps / self.target_fps)
            self.effective_fps = self.original_fps / self.frame_skip
        else:
            self.frame_skip = 1
            self.effective_fps = self.original_fps

        self.frame_map = list(range(0, self.total_frames, self.frame_skip))

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.path)
        frame_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
        ratio = self.input_size / max(frame_size)
        new_size = tuple([int(x * ratio) for x in frame_size])
        delta_w = self.input_size - new_size[1]
        delta_h = self.input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        images = []
        frame_idx = 0
        while True:
            ret, img = cap.read()
            if not ret or img is None:
                break
            if frame_idx % self.frame_skip == 0:
                resized = cv2.resize(img, (new_size[1], new_size[0]))
                b_img = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                           cv2.BORDER_CONSTANT,
                                           value=[0.406 * 255, 0.456 * 255, 0.485 * 255])
                b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
                images.append(b_img_rgb)
            frame_idx += 1
        cap.release()

        labels = np.zeros(len(images))
        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor:
    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images = images.transpose((0, 3, 1, 2))
        return {'images': torch.from_numpy(images).float().div(255.),
                'labels': torch.from_numpy(labels).long()}


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return {'images': images, 'labels': labels}


def load_models():
    """Load both SwingNet and CBAM-SwingNet models."""
    global swingnet_model, cbam_model, device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading models on {device}...")

    # SwingNet
    swingnet_model = EventDetector(
        pretrain='mobilenet_v2.pth.tar',
        width_mult=1.,
        lstm_layers=1,
        lstm_hidden=256,
        bidirectional=True,
        dropout=False
    )
    save_dict = torch.load('models/swingnet_1800.pth.tar', map_location=device)
    swingnet_model.load_state_dict(save_dict['model_state_dict'])
    swingnet_model.to(device)
    swingnet_model.eval()
    print("SwingNet loaded.")

    # CBAM-SwingNet
    cbam_model = CBAMEventDetector(
        pretrain_path=None,
        width_mult=1.,
        lstm_layers=1,
        lstm_hidden=256,
        bidirectional=True,
        dropout=False
    )
    checkpoint = torch.load('cbam_swingnet/models/cbam_swingnet.pth.tar', map_location=device)
    if 'model_state_dict' in checkpoint:
        cbam_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        cbam_model.load_state_dict(checkpoint)
    cbam_model.to(device)
    cbam_model.eval()
    print("CBAM-SwingNet loaded.")


def run_inference(model, video_path, seq_length=64, temporal_order=False):
    """Run inference on a video with a given model."""
    ds = VideoDataset(
        video_path,
        transform=transforms.Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    for sample in dl:
        images = sample['images']
        batch = 0
        probs = None

        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]

            image_batch = image_batch.to(device)
            with torch.no_grad():
                logits = model(image_batch)

            if probs is None:
                probs = torch.softmax(logits, dim=1)
            else:
                probs = torch.cat((probs, torch.softmax(logits, dim=1)), 0)
            batch += 1

    confidence = probs.cpu().numpy()

    if temporal_order:
        sampled_events = np.zeros(8, dtype=int)
        min_frame = 0
        for i in range(8):
            search_probs = confidence[min_frame:, i]
            if len(search_probs) > 0:
                best_in_window = np.argmax(search_probs)
                sampled_events[i] = min_frame + best_in_window
                min_frame = sampled_events[i]
            else:
                sampled_events[i] = min_frame
    else:
        sampled_events = np.argmax(confidence, axis=0)[:-1]

    # Map back to original frame numbers
    original_events = []
    for e in sampled_events:
        if e < len(ds.frame_map):
            original_events.append(ds.frame_map[e])
        else:
            original_events.append(ds.frame_map[-1])
    original_events = np.array(original_events)

    # Extract frame images as base64
    cap = cv2.VideoCapture(video_path)
    fps = ds.original_fps
    frame_images = []
    for i, orig_frame in enumerate(original_events):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(orig_frame))
        ret, frame = cap.read()
        if ret:
            label = f"{EVENT_NAMES[i]} (Frame {orig_frame})"
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {orig_frame/fps:.3f}s", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_images.append(base64.b64encode(buffer).decode('utf-8'))
        else:
            frame_images.append(None)
    cap.release()

    # Build results
    events = []
    for i, (sampled_e, orig_e) in enumerate(zip(sampled_events, original_events)):
        conf = float(confidence[sampled_e, i])
        events.append({
            'event_id': i,
            'event_name': EVENT_NAMES[i],
            'frame': int(orig_e),
            'time_seconds': float(orig_e / fps),
            'confidence': conf,
            'frame_image': frame_images[i]
        })

    avg_conf = float(np.mean([confidence[sampled_events[i], i] for i in range(8)]))
    backswing = float((original_events[3] - original_events[0]) / fps) if fps > 0 else 0
    downswing = float((original_events[5] - original_events[3]) / fps) if fps > 0 else 0

    return {
        'video_info': {
            'fps': float(fps),
            'original_fps': float(ds.original_fps),
            'effective_fps': float(ds.effective_fps),
            'frame_count': ds.total_frames,
            'frame_skip': ds.frame_skip,
        },
        'events': events,
        'metrics': {
            'average_confidence': avg_conf,
            'backswing_duration': backswing,
            'downswing_duration': downswing,
        }
    }


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# ---- Routes ----

@app.route('/')
def index():
    """Serve the upload/analysis page."""
    return send_from_directory('.', 'upload.html')


@app.route('/dashboard')
def dashboard():
    """Serve the comparison dashboard."""
    return send_from_directory('comparison', 'index.html')


@app.route('/comparison/<path:filename>')
def comparison_files(filename):
    """Serve comparison static files (images, etc.)."""
    return send_from_directory('comparison', filename)


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'swingnet_loaded': swingnet_model is not None,
        'cbam_loaded': cbam_model is not None,
        'device': str(device)
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze a golf swing video."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {ALLOWED_EXTENSIONS}'}), 400

    model_name = request.form.get('model', 'cbam')
    temporal = request.form.get('temporal_order', 'true').lower() == 'true'

    if model_name == 'swingnet':
        model = swingnet_model
    else:
        model = cbam_model

    temp_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.mp4")
    try:
        file.save(temp_path)
        results = run_inference(model, temp_path, temporal_order=temporal)
        results['model'] = model_name
        results['temporal_order'] = temporal
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# Load models at import time (works with gunicorn preload_app=True)
load_models()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"\nGolf Swing Analysis App running on port {port}")
    print(f"  Upload & Analyze: http://localhost:{port}/")
    print(f"  Dashboard:        http://localhost:{port}/dashboard")
    app.run(host='0.0.0.0', port=port, debug=False)
