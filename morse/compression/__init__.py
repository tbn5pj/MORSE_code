"""Scientific compressor implementations."""
from .one_pass import compress_one_candidate
from .isd import compress_isd_candidate

__all__ = ["compress_one_candidate", "compress_isd_candidate"]

