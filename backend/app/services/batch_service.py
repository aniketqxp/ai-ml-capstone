import os

class BatchFileService:
    def __init__(self, watch_dir: str = "/data"):
        self.watch_dir = watch_dir

    def scan_records(self):
        # Reads local AppTek records and maps metadata keys from disk
        if not os.path.exists(self.watch_dir):
            return []
        return os.listdir(self.watch_dir)
