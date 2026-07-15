import cv2
import torch
import numpy as np

from model import LipNet

# ------------------
# Character Mapping
# ------------------

vocab = "abcdefghijklmnopqrstuvwxyz "

char_to_num = {
    c: i + 1 for i, c in enumerate(vocab)
}

num_to_char = {
    i + 1: c for i, c in enumerate(vocab)
}

# ------------------
# Load Model
# ------------------

# ------------------
# Load Model
# ------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = LipNet().to(device)

model.load_state_dict(
    torch.load(
        "best_lipnet.pth",
        map_location=device
    )
)

model.eval()

print("Using Device:", device)

# ------------------
# Extract Lips
# ------------------

def extract_lips(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        h, w, _ = frame.shape

        mouth = frame[
            int(h * 0.55):int(h * 0.85),
            int(w * 0.25):int(w * 0.75)
        ]

        mouth = cv2.resize(
            mouth,
            (112, 112)
        )

        frames.append(mouth)

    cap.release()

    return np.array(frames)

# ------------------
# Predict
# ------------------

def predict_video(video_path):

    frames = extract_lips(video_path)

    print("=" * 80)
    print("Video:", video_path)
    print("Frames Shape:", frames.shape)

    if len(frames) == 0:
        return "No frames found."

    print("Pixel Mean:", np.mean(frames))
    print("Pixel Std :", np.std(frames))

    frames = frames[:75]

    if len(frames) < 75:

        pad = np.zeros(
            (
                75 - len(frames),
                112,
                112,
                3
            ),
            dtype=np.uint8
        )

        frames = np.concatenate(
            [frames, pad],
            axis=0
        )

    frames = frames.astype(np.float32) / 255.0

    x = torch.tensor(frames).unsqueeze(0)

    print("Input Shape:", x.shape)

    with torch.no_grad():

        output = model(x)

        print("Output Shape:", output.shape)

        print("\nFirst Frame Logits")
        print(output[0, 0, :10])

        print("\nMiddle Frame Logits")
        print(output[0, 35, :10])

        print("\nLast Frame Logits")
        print(output[0, -1, :10])

        pred = output.argmax(dim=2)[0]

        print("\nPredicted Indices")
        print(pred.tolist())

    text = ""

    previous = -1

    for idx in pred:

        idx = idx.item()

        if idx == previous:
            continue

        previous = idx

        if idx == 0:
            continue

        text += num_to_char.get(idx, "")

    print("\nPrediction:", text)
    print("=" * 80)

    return text.strip()