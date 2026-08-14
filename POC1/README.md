# Live age + gender (local)

InsightFace face detect → FairFace age + gender.

No API.

## Models

- Face: InsightFace `buffalo_l` (RetinaFace), frontal + near only
- Age: [dima806/fairface_age_image_detection](https://huggingface.co/dima806/fairface_age_image_detection)
- Gender: [dima806/fairface_gender_image_detection](https://huggingface.co/dima806/fairface_gender_image_detection) (~93%)

## Run

```bash
cd /home/webclues-nidhi/mYpY/RnD/Age_detectoion/live
source ../env/bin/activate

python run.py --source /path/to/image.jpg --no-display
python run.py --source /path/to/images/ --no-display
python run.py --source /path/to/video.mp4
python run.py --source rtsp://USER:PASS@HOST:554/stream1
```

Output: `live/output/`
