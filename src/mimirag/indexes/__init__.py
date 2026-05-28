"""Vector index backends. FAISS-CPU is the default."""

from mimirag.indexes.faiss_index import FaissCpuIndex

__all__ = ["FaissCpuIndex"]
