"""
Golf Swing Analysis API Server
Accepts video uploads and returns detected swing events with frame timestamps.
"""

import os
import tempfile
import uuid
from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import EventDetector

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'm4v', 'MP4', 'MOV'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Event names
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

# Global model (loaded once at startup)
model = None
device = None


class VideoDataset(Dataset):
    """Dataset for processing uploaded video."""
    def __init__(self, path, input_size=160, transform=None):
        self.path = path
        self.input_size = input_size
        self.transform = transform

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
        for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, img = cap.read()
            if img is None:
                break
            resized = cv2.resize(img, (new_size[1], new_size[0]))
            b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=[0.406 * 255, 0.456 * 255, 0.485 * 255])
            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            images.append(b_img_rgb)
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


def load_model():
    """Load the SwingNet model."""
    global model, device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on {device}...")

    model = EventDetector(
        pretrain='mobilenet_v2.pth.tar',
        width_mult=1.,
        lstm_layers=1,
        lstm_hidden=256,
        bidirectional=True,
        dropout=False
    )

    try:
        save_dict = torch.load('models/swingnet_1800.pth.tar', map_location=device)
    except FileNotFoundError:
        save_dict = torch.load('swingnet_1800.pth.tar', map_location=device)

    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully!")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def analyze_video(video_path, seq_length=64):
    """Run swing detection on a video file."""
    global model, device

    # Get video info
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    # Create dataset and dataloader
    ds = VideoDataset(video_path, transform=transforms.Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    # Run inference
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

    # Get event predictions
    confidence_matrix = probs.cpu().numpy()
    events = np.argmax(confidence_matrix, axis=0)[:-1]  # exclude "no event" class

    # Build results
    results = {
        'video_info': {
            'fps': fps,
            'frame_count': frame_count,
            'duration_seconds': duration
        },
        'events': []
    }

    for i, frame_num in enumerate(events):
        event = {
            'event_id': i,
            'event_name': EVENT_NAMES[i],
            'frame': int(frame_num),
            'time_seconds': float(frame_num / fps) if fps > 0 else 0,
            'confidence': float(confidence_matrix[frame_num, i])
        }
        results['events'].append(event)

    # Calculate swing metrics
    if len(events) == 8:
        results['swing_metrics'] = {
            'backswing_duration': float((events[3] - events[0]) / fps) if fps > 0 else 0,  # Address to Top
            'downswing_duration': float((events[5] - events[3]) / fps) if fps > 0 else 0,  # Top to Impact
            'total_swing_duration': float((events[7] - events[0]) / fps) if fps > 0 else 0,  # Address to Finish
            'average_confidence': float(np.mean([confidence_matrix[events[i], i] for i in range(8)]))
        }

    return results


def extract_frame(video_path, frame_num):
    """Extract a single frame from the video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if ret:
        return frame
    return None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else None
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze a golf swing video.

    Expects: multipart/form-data with 'video' file
    Returns: JSON with detected events and timing
    """
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {ALLOWED_EXTENSIONS}'}), 400

    # Save uploaded file temporarily
    temp_filename = f"{uuid.uuid4()}.mp4"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)

    try:
        file.save(temp_path)

        # Analyze the video
        results = analyze_video(temp_path)

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/analyze-with-frames', methods=['POST'])
def analyze_with_frames():
    """
    Analyze a golf swing video and return event frames as base64 images.

    Expects: multipart/form-data with 'video' file
    Returns: JSON with detected events, timing, and base64-encoded frame images
    """
    import base64

    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {ALLOWED_EXTENSIONS}'}), 400

    temp_filename = f"{uuid.uuid4()}.mp4"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)

    try:
        file.save(temp_path)

        # Analyze the video
        results = analyze_video(temp_path)

        # Extract and encode frames for each event
        for event in results['events']:
            frame = extract_frame(temp_path, event['frame'])
            if frame is not None:
                # Add label to frame
                fps = results['video_info']['fps']
                label = f"{event['event_name']} (Frame {event['frame']})"
                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Time: {event['time_seconds']:.3f}s", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Encode as JPEG base64
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                event['frame_image_base64'] = base64.b64encode(buffer).decode('utf-8')

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/extract-frame/<int:frame_num>', methods=['POST'])
def get_frame(frame_num):
    """
    Extract a specific frame from an uploaded video.

    Expects: multipart/form-data with 'video' file
    Returns: JPEG image
    """
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    temp_filename = f"{uuid.uuid4()}.mp4"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)

    try:
        file.save(temp_path)
        frame = extract_frame(temp_path, frame_num)

        if frame is None:
            return jsonify({'error': f'Could not extract frame {frame_num}'}), 400

        # Encode and return as JPEG
        _, buffer = cv2.imencode('.jpg', frame)

        return send_file(
            io.BytesIO(buffer),
            mimetype='image/jpeg',
            as_attachment=False
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == '__main__':
    import io

    # Load model at startup
    load_model()

    # Run server
    print("\n" + "="*50)
    print("Golf Swing Analysis API Server")
    print("="*50)
    print("\nEndpoints:")
    print("  GET  /health              - Health check")
    print("  POST /analyze             - Analyze video, return JSON")
    print("  POST /analyze-with-frames - Analyze video, return JSON with base64 frames")
    print("\nExample usage:")
    print("  curl -X POST -F 'video=@swing.mp4' http://localhost:5000/analyze")
    print("="*50 + "\n")

    app.run(host='0.0.0.0', port=8080, debug=False)
