SAMPLE_RATE = 44_100
BIT_DEPTH   = "PCM_16"
FADE_SECS   = 2.0

HIRES_SAMPLE_RATE = 48_000
HIRES_BIT_DEPTH   = "PCM_24"


def set_hires():
    """Switch to 48kHz/24-bit for the current process."""
    global SAMPLE_RATE, BIT_DEPTH
    SAMPLE_RATE = HIRES_SAMPLE_RATE
    BIT_DEPTH   = HIRES_BIT_DEPTH
