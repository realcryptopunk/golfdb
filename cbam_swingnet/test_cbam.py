"""
Test CBAM-SwingNet on golf swing videos
With automatic frame rate normalization for high-fps videos
"""

import argparse
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from cbam_model import CBAMEventDetector

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

TARGET_FPS = 30  # Model was trained on ~30fps videos


class SampleVideo(Dataset):
    """
    Video dataset with automatic frame rate normalization.

    For high-fps videos (>45fps), samples frames to approximate 30fps
    to match the training data distribution.
    """
    def __init__(self, path, input_size=160, transform=None, target_fps=TARGET_FPS):
        self.path = path
        self.input_size = input_size
        self.transform = transform
        self.target_fps = target_fps

        # Get video properties
        cap = cv2.VideoCapture(path)
        self.original_fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Calculate frame skip for high-fps videos
        if self.original_fps > 45:  # High fps threshold
            self.frame_skip = round(self.original_fps / self.target_fps)
            self.effective_fps = self.original_fps / self.frame_skip
        else:
            self.frame_skip = 1
            self.effective_fps = self.original_fps

        # Store frame mapping (sampled index -> original frame number)
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

            # Only process frames according to frame_skip
            if frame_idx % self.frame_skip == 0:
                resized = cv2.resize(img, (new_size[1], new_size[0]))
                b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
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


def extract_event_frames(video_path, events, original_events, output_dir='event_frames_cbam'):
    """
    Extract and save frames for each detected event.

    Args:
        video_path: Path to original video
        events: Event frame numbers (in sampled space)
        original_events: Event frame numbers mapped back to original video
        output_dir: Output directory for saved frames
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    saved_frames = []
    for i, (sampled_frame, orig_frame) in enumerate(zip(events, original_events)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, orig_frame)
        ret, frame = cap.read()
        if ret:
            label = f"{EVENT_NAMES[i]} (Frame {orig_frame})"
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {orig_frame/fps:.3f}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            filename = f"{i+1}_{EVENT_NAMES[i].replace('-', '_').replace(' ', '_')}_frame{orig_frame}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_frames.append(filepath)
            print(f"  Saved: {filepath}")

    cap.release()
    return saved_frames


def main():
    parser = argparse.ArgumentParser(description='CBAM-SwingNet Golf Swing Analyzer')
    parser.add_argument('-p', '--path', required=True, help='Path to video file')
    parser.add_argument('-s', '--seq-length', type=int, default=64, help='Sequence length')
    parser.add_argument('--save-frames', action='store_true', help='Save event frame screenshots')
    parser.add_argument('--output-dir', type=str, default='event_frames_cbam', help='Output directory')
    parser.add_argument('--weights', type=str, default='models/cbam_swingnet.pth.tar', help='Path to model weights')
    parser.add_argument('--target-fps', type=int, default=30, help='Target FPS for normalization (default: 30)')
    parser.add_argument('--no-fps-norm', action='store_true', help='Disable automatic FPS normalization')
    parser.add_argument('--temporal-order', action='store_true', help='Enforce temporal ordering of events')
    args = parser.parse_args()

    seq_length = args.seq_length
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('='*60)
    print('CBAM-SwingNet Golf Swing Analyzer')
    print('='*60)
    print(f'Device: {device}')
    print(f'Video: {args.path}')
    print()

    # Load model
    print('Loading CBAM-SwingNet model...')
    model = CBAMEventDetector(
        pretrain_path=None,
        width_mult=1.,
        lstm_layers=1,
        lstm_hidden=256,
        bidirectional=True,
        dropout=False
    )

    # Load trained weights
    checkpoint = torch.load(args.weights, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print('Model loaded successfully!')
    print()

    # Create dataset with FPS normalization
    target_fps = None if args.no_fps_norm else args.target_fps
    ds = SampleVideo(
        args.path,
        transform=transforms.Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        target_fps=target_fps if target_fps else 999  # Large number = no skipping
    )

    # Show FPS info
    print(f'Original FPS: {ds.original_fps:.2f}')
    if ds.frame_skip > 1:
        print(f'Frame skip: {ds.frame_skip}x (processing every {ds.frame_skip} frames)')
        print(f'Effective FPS: {ds.effective_fps:.2f}')
        print(f'Frames to process: {len(ds.frame_map)} (of {ds.total_frames} total)')
    else:
        print(f'No frame skipping needed (FPS within normal range)')
    print()

    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    # Process video
    print('Analyzing video...')
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

    # Get predictions (in sampled frame space)
    confidence = probs.cpu().numpy()
    num_frames = confidence.shape[0]

    if args.temporal_order:
        # Temporal ordering: each event must be at or after the previous event's frame
        sampled_events = np.zeros(8, dtype=int)
        min_frame = 0
        for i in range(8):
            # Search from min_frame onward for the peak of event class i
            search_probs = confidence[min_frame:, i]
            if len(search_probs) > 0:
                best_in_window = np.argmax(search_probs)
                sampled_events[i] = min_frame + best_in_window
                min_frame = sampled_events[i]  # next event must be at or after this
            else:
                sampled_events[i] = min_frame
    else:
        # Original: independent global peak per event class
        sampled_events = np.argmax(confidence, axis=0)[:-1]

    # Map back to original frame numbers
    original_events = []
    for e in sampled_events:
        if e < len(ds.frame_map):
            original_events.append(ds.frame_map[e])
        else:
            original_events.append(ds.frame_map[-1])
    original_events = np.array(original_events)

    # Print results
    print()
    print('='*60)
    mode_label = 'TEMPORAL' if args.temporal_order else 'INDEPENDENT'
    print(f'DETECTED SWING EVENTS (CBAM-SwingNet, {mode_label})')
    print('='*60)

    for i, (sampled_e, orig_e) in enumerate(zip(sampled_events, original_events)):
        conf = confidence[sampled_e, i] * 100
        time_sec = orig_e / ds.original_fps
        print(f'{EVENT_NAMES[i]:20s}: Frame {orig_e:4d} | {time_sec:6.3f}s | {conf:5.1f}% conf')

    print('='*60)
    print(f'Original Video FPS: {ds.original_fps:.2f}')
    if ds.frame_skip > 1:
        print(f'Analysis FPS: {ds.effective_fps:.2f} (normalized)')

    # Calculate metrics
    avg_conf = np.mean([confidence[sampled_events[i], i] for i in range(8)]) * 100
    print(f'Average Confidence: {avg_conf:.1f}%')

    if original_events[3] > original_events[0]:  # Top after Address
        backswing = (original_events[3] - original_events[0]) / ds.original_fps
        print(f'Backswing Duration: {backswing:.3f}s')

    if original_events[5] > original_events[3]:  # Impact after Top
        downswing = (original_events[5] - original_events[3]) / ds.original_fps
        print(f'Downswing Duration: {downswing:.3f}s')

    print('='*60)

    # Save frames if requested
    if args.save_frames:
        print(f'\nSaving event frames to: {args.output_dir}/')
        saved = extract_event_frames(args.path, sampled_events, original_events, args.output_dir)
        print(f'Saved {len(saved)} frames.')


if __name__ == '__main__':
    main()
