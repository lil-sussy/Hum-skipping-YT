
import os


# where to store cached audio, deleted frequently
CONFIG_VIDEO_CACHE = os.path.expanduser("~/.cache/humaware-yt-skip/videos/")

# where to store processed classifications json
CONFIG_CLASSIFICATIONS_CACHE = os.path.expanduser("~/.cache/humaware-yt-skip/classifications/")

# Length of each frame served
# CONFIG_CHOSEN_FRAME_LENGTHS_SEC = [0.3, 0.5, 1, 2, 4] # Initially tried and obscurely worked, to this day I don't know why
CONFIG_CHOSEN_FRAME_LENGTHS_SEC = [0.032]

# the hop size of the classifier, higher the more accurate but takes longer (= frame length / HOP_RATIO)
HOP_RATIO = 2

# it is not the treshold for the classifier, but how much tolerance to allow when merging segments, i dont know what it is but is in seconds
CONFIG_MERGING_TOLERANCE = 0.15

# model path
CONFIG_MODEL_PATH = "./src/HumAware-VAD/HumAwareVAD.jit"

# audio sample rate required by the model, defined here in config but should't be changed
CONFIG_AUDIO_SR = 16000 # or 8000