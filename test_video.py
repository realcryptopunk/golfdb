import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import EventDetector

event_names = {
    0: 'Address',
    1: 'Toe-up',
    2: 'Mid-backswing',
    3: 'Top',
    4: 'Mid-downswing',
    5: 'Impact',
    6: 'Mid-follow-through',
    7: 'Finish'
}


class SampleVideo(Dataset):
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

        # preprocess and store frames
        images = []
        for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, img = cap.read()
            if img is None:
                break
            resized = cv2.resize(img, (new_size[1], new_size[0]))
            b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=[0.406 * 255, 0.456 * 255, 0.485 * 255])  # RGB mean
            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            images.append(b_img_rgb)
        cap.release()
        labels = np.zeros(len(images))  # placeholder
        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images = images.transpose((0, 3, 1, 2))
        return {'images': torch.from_numpy(images).float().div(255.),
                'labels': torch.from_numpy(labels).long()}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return {'images': images, 'labels': labels}


def extract_event_frames(video_path, events, event_names, output_dir='event_frames'):
    """Extract and save frames for each detected event."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    saved_frames = []
    for i, frame_num in enumerate(events):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            # Add event label to the frame
            label = f"{event_names[i]} (Frame {frame_num})"
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {frame_num/fps:.3f}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Save frame
            filename = f"{i+1}_{event_names[i].replace('-', '_').replace(' ', '_')}_frame{frame_num}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_frames.append(filepath)
            print(f"  Saved: {filepath}")

    cap.release()
    return saved_frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Path to video', required=True)
    parser.add_argument('-s', '--seq-length', type=int, help='Number of frames', default=64)
    parser.add_argument('--no-display', action='store_true', help='Do not display video')
    parser.add_argument('--save-frames', action='store_true', help='Save screenshots of event frames')
    parser.add_argument('--output-dir', type=str, default='event_frames', help='Output directory for saved frames')
    args = parser.parse_args()

    seq_length = args.seq_length

    print('Loading model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = EventDetector(pretrain='mobilenet_v2.pth.tar',
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)

    try:
        save_dict = torch.load('models/swingnet_1800.pth.tar', map_location=device)
    except FileNotFoundError:
        save_dict = torch.load('swingnet_1800.pth.tar', map_location=device)
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    print('Model loaded.')

    print(f'\nAnalyzing video: {args.path}')
    ds = SampleVideo(args.path, transform=transforms.Compose([ToTensor(),
                                                               Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])]))
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    for sample in dl:
        images = sample['images']
        # full video inference
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
            image_batch = image_batch.to(device)

            with torch.no_grad():
                logits = model(image_batch)

            if batch == 0:
                probs = torch.softmax(logits, dim=1)
            else:
                probs = torch.cat((probs, torch.softmax(logits, dim=1)), 0)
            batch += 1

    events = np.argmax(probs.cpu().numpy(), axis=0)[:-1]  # exclude last class (no event)
    print('\n' + '='*50)
    print('DETECTED SWING EVENTS')
    print('='*50)

    confidence = probs.cpu().numpy()
    for i, e in enumerate(events):
        conf = confidence[e, i] * 100
        print(f'{event_names[i]:20s}: Frame {e:4d} (confidence: {conf:.1f}%)')

    print('='*50)

    cap = cv2.VideoCapture(args.path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print(f'\nVideo FPS: {fps:.2f}')
    print('\nEvent Timing:')
    print('-'*50)
    for i, e in enumerate(events):
        time_sec = e / fps
        print(f'{event_names[i]:20s}: {time_sec:.3f}s')

    # Save event frame screenshots
    if args.save_frames:
        print(f'\nSaving event frame screenshots to: {args.output_dir}/')
        saved = extract_event_frames(args.path, events, event_names, args.output_dir)
        print(f'\nSaved {len(saved)} event frames.')

    # Optionally display video with annotations
    if not args.no_display:
        try:
            cap = cv2.VideoCapture(args.path)
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Add event annotation if this is an event frame
                for i, e in enumerate(events):
                    if e == frame_idx:
                        cv2.putText(frame, event_names[i], (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('Golf Swing Analysis', frame)
                if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                    break
                frame_idx += 1
            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f'\nCould not display video: {e}')
            print('Use --no-display flag if running in a headless environment.')
