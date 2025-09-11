# dspy.configure_cache

## dspy.configure_cache

```python
def configure_cache(enable_disk_cache=True, enable_memory_cache=True, disk_cache_dir=DISK_CACHE_DIR, disk_size_limit_bytes=DISK_CACHE_LIMIT, memory_max_entries=1000000)
```

Configure the cache for DSPy.

Args:
    enable_disk_cache: Whether to enable on-disk cache.
    enable_memory_cache: Whether to enable in-memory cache.
    disk_cache_dir: The directory to store the on-disk cache.
    disk_size_limit_bytes: The size limit of the on-disk cache.
    memory_max_entries: The maximum number of entries in the in-memory cache.

Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/clients/__init__.py` (lines 17–44)

