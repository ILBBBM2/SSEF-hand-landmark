# Sign-language landmark translator

This project uses Google's MediaPipe Hand Landmarker to track up to two webcam hands,
then trains a small PyTorch classifier on their landmarks. It is
designed for *static* signs first (letters, numbers, and fixed words). Dynamic
signs and full sign-language grammar need a sequence model and a larger,
language-specific dataset.

## Workflow

### Upgrade an existing one-hand dataset

The default two-hand cache is `two_hand_landmark_dataset`. To keep your old
one-hand recordings, migrate them once; the original `landmark_dataset` is not
modified. Each old sample becomes both a left-only and right-only training
sample:

```powershell
python migrate_to_two_hand_dataset.py
```

Then collect new two-hand labels into the default output, and train a fresh
model. One-hand signs remain supported; their missing hand is represented by
an empty slot.

1. Build a dataset from labeled videos. The included `vids/` folder can contain
   one video per label, named `A.mov`, `B.mov`, `0.mov`, and so on. Only frames
   where MediaPipe finds a hand are saved:

   ```powershell
   python extract_video_landmarks.py --input .\vids --sample-fps 6
   ```

   Add `--visualize` to view each sampled frame with Google's official
   MediaPipe landmark skeleton and left/right handedness label overlaid. Green
   means a hand was saved; red means no hand was detected.
   Press `Q` to close just the preview while extraction continues:

   ```powershell
   python extract_video_landmarks.py --input .\vids --sample-fps 6 --visualize --output .\video_preview_dataset
   ```

   Use a separate `--output` folder when you only want to preview videos; this
   prevents the preview extraction from replacing webcam samples in
   `landmark_dataset`. Portrait 4K videos are automatically scaled to fit the
   preview window without cropping; landmark extraction still uses the original
   full-resolution frames.

   Or record one label at a time from the webcam:

   ```powershell
   python collect_landmarks.py --label A --samples 300
   python collect_landmarks.py --label B --samples 300
   ```

   The collector displays the same live skeleton overlay. Press `R` to
   start/pause saving samples and `Q` to finish the current label. It appends
   samples to `landmark_dataset` by default; use `--output` to create a
   separate custom dataset.

   **VS Code:** choose **Run and Debug** in the left sidebar, select
   **Collect webcam dataset (asks for label)** in the launch dropdown, then
   press the green Play button (or `F5`). Type the label into the integrated
   terminal before the webcam window opens.

   To remove a label you recorded incorrectly, run:

   ```powershell
   python delete_landmark_label.py --label A
   ```

   It tells you how many samples will be removed and requires you to type
   `DELETE A` before changing the dataset. A backup is saved in
   `landmark_dataset/backups` first. The matching VS Code Play configuration is
   **Delete a webcam dataset label (asks for label)**. Retrain afterward.

   Record several short takes per label, varying distance, lighting, and hand angle.
   Keep both hands and wrists in frame. Use the same label spelling every time.

2. Train the landmark classifier:

   ```powershell
   python train.py --model landmark_mlp --fresh --device cuda
   ```

   The project automatically uses CUDA when it is available. `--device cuda`
   requires the GPU explicitly, while `--device auto` (the default) falls back
   to CPU when CUDA is unavailable. In VS Code, use **Train landmark classifier
   (GPU)** from the Run and Debug dropdown. Training saves a checkpoint only
   when validation accuracy improves, and stops after eight non-improving
   epochs. Change this with `--patience`; use `--patience 0` to disable early
   stopping.

3. Evaluate the saved best checkpoint on the reproducible validation split:

   ```powershell
   python evaluate.py --model landmark_mlp
   ```

   This writes `evaluation/metrics.json`, per-class precision/recall/F1, a
   numeric confusion matrix, and `evaluation/confusion_matrix.png`. The default
   split seed is 42; use the same `--seed` during training and evaluation when
   changing it.

4. Translate from the webcam:

   ```powershell
   python predict_webcam.py
   ```

   The display smooths predictions. Press `Space` to add the stable sign to the
   translation, `Backspace` to remove a word, `C` to clear, and `Q` to quit.

`extract_landmarks.py` remains available only to import an existing ImageFolder
dataset. For the supplied videos, use `extract_video_landmarks.py`.
