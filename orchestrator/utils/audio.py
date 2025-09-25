import numpy as np


def resample_pcm(pcm_bytes: bytes, src_frame_rate: int, dst_frame_rate: int) -> bytes:
    """Resample PCM audio data to the target frame rate.

    This function converts PCM audio data from one sample rate to another using
    linear interpolation. The input PCM data is expected to be 16-bit signed
    integers in little-endian byte order.

    Args:
        pcm_bytes (bytes):
            The input PCM audio data as bytes.
        src_frame_rate (int):
            The source sample rate in Hz.
        dst_frame_rate (int):
            The target sample rate in Hz.

    Returns:
        bytes:
            The resampled PCM audio data as bytes with the target sample rate.
    """
    pcm_data = np.frombuffer(pcm_bytes, dtype=np.int16)
    num_samples = len(pcm_data)
    num_target_samples = int(num_samples * dst_frame_rate / src_frame_rate)
    resampled_data = np.interp(
        np.linspace(0, num_samples, num_target_samples, endpoint=False), np.arange(num_samples), pcm_data
    )

    # Ensure data is in little-endian byte order
    if resampled_data.dtype.byteorder == ">":
        resampled_data = resampled_data.byteswap()

    return resampled_data.astype(np.int16).tobytes()
