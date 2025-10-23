
# where to store cached audio, deleted frequently
CONFIG_VIDEO_CACHE = "~/.cache/humaware-yt-skip/videos/"

# where to store processed classifications json
CONFIG_CLASSIFICATIONS_CACHE = "~/.cache/humaware-yt-skip/classifications/"

# Length of each frame served
CONFIG_CHOSEN_FRAME_LENGTHS_SEC = [0.3, 0.5, 1, 2, 4]

# the hop size of the classifier, higher the more accurate but takes longer (= frame length / HOP_RATIO)
HOP_RATIO = 15

# model path
CONFIG_MODEL_PATH = "HumAware-VAD/HumAwareVAD.jit"