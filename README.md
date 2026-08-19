# Sign-language landmark translator

This project uses Google's MediaPipe Hand Landmarker to track a webcam hand,
then trains a small PyTorch classifier on the 21 detected landmarks. It is
designed for *static* signs first (letters, numbers, and fixed words). Dynamic
signs and full sign-language grammar need a sequence model and a larger,
language-specific dataset.

## Workflow

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
   Keep the full hand and wrist in frame. Use the same label spelling every time.

2. Train the landmark classifier:

   ```powershell
   python train.py --model landmark_mlp --fresh --device cuda
   ```

   The project automatically uses CUDA when it is available. `--device cuda`
   requires the GPU explicitly, while `--device auto` (the default) falls back
   to CPU when CUDA is unavailable. In VS Code, use **Train landmark classifier
   (GPU)** from the Run and Debug dropdown.

3. Translate from the webcam:

   ```powershell
   python predict_webcam.py
   ```

   The display smooths predictions. Press `Space` to add the stable sign to the
   translation, `Backspace` to remove a word, `C` to clear, and `Q` to quit.

`extract_landmarks.py` remains available only to import an existing ImageFolder
dataset. For the supplied videos, use `extract_video_landmarks.py`.
