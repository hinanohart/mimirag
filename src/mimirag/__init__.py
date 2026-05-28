"""mimirag: audio-native RAG over Kyutai Mimi 12.5 Hz semantic tokens.

Pre-alpha. See README.md for the CLAIM table — implementation novelty is
explicit, performance claims require `bench/RESULTS.md` evidence.
"""

from mimirag._version import __version__
from mimirag.models import AudioChunk, Hit, RetrievalResult, TokenStream
from mimirag.protocols import Encoder, Fuser, IndexBackend

__all__ = [
    "AudioChunk",
    "Encoder",
    "Fuser",
    "Hit",
    "IndexBackend",
    "RetrievalResult",
    "TokenStream",
    "__version__",
]
