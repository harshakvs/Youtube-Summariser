# Video Summarizer → Talking-Head Shorts

Turn a long YouTube video into a short clip of the speaker *saying* the summary.

Paste a video URL and the app transcribes it with Whisper, summarizes the transcript, converts the summary to speech, finds the clearest frame of the speaker's face, and uses Wav2Lip to lip-sync that face to the new audio. The result is a short, watchable recap instead of a wall of text.

> Built in 2024 on top of [somanyadav/Youtube-Summariser](https://github.com/somanyadav/Youtube-Summariser), which provides the text summarization, translation, and text-to-speech. The speech recognition, face detection, lip-sync video generation, and multi-page app flow are my additions.

## How it works

```
YouTube URL
   │
   ├─▶ download video + audio (pytube, moviepy)
   │
   ├─▶ transcribe ─────────── Whisper large-v3 (FP16 on GPU, 30s chunks, batched)
   │
   ├─▶ summarize ─────────── extractive (Gensim, NLTK, spaCy, TF-IDF) or abstractive (T5, BART)
   │        └─▶ translate + text-to-speech (Google Translate, gTTS)
   │
   ├─▶ best face frame ───── OpenCV DNN face detector (SSD ResNet-10), keeps the largest confident face
   │
   └─▶ lip-sync ──────────── Wav2Lip (GAN checkpoint) drives the face with the summary audio
                                  │
                                  ▼
                        short talking-head video
```

## What I added

| Piece | What it does |
|---|---|
| **Whisper transcription** | Replaces YouTube's captions with `openai/whisper-large-v3` via Hugging Face, so videos without captions work too |
| **Face selection** (`face.py`) | Scans frames with an SSD face detector and keeps the largest high-confidence (>0.8) face as the lip-sync source |
| **Lip-sync generation** | Runs Wav2Lip on the chosen face and the summary audio to produce the final clip |
| **Media utilities** | Download, trim, convert and merge audio/video (`download.py`, `cut-length.py`, `audio-check.py`) |
| **App flow** | Three Streamlit pages: download + face detection → transcript → generated short |

## Running it

This is a research prototype written on Windows, so file paths are hard-coded in `app.py` and the helper scripts. Update them for your machine first.

1. `pip install -r requirements.txt streamlit moviepy pydub opencv-python`
2. Clone [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) and download its `wav2lip_gan.pth` checkpoint
3. Point the Wav2Lip paths in `app.py` at your clone
4. `streamlit run app.py`

A GPU is strongly recommended for Whisper large-v3 and Wav2Lip.

## Responsible use

Lip-sync makes it easy to put words in someone's mouth. Use this only on content you own or have permission to adapt, and label generated clips as AI-generated.

## Credits

- Summarization app: [somanyadav/Youtube-Summariser](https://github.com/somanyadav/Youtube-Summariser)
- Lip-sync: [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) (Prajwal et al., ACM MM 2020)
- Speech recognition: [Whisper](https://github.com/openai/whisper) (OpenAI)
