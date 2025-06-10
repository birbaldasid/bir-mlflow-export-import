import threading
import time
from datetime import datetime
import os
import pandas as pd
import logging
import pyarrow.dataset as ds
from mlflow_export_import.common import utils
from mlflow_export_import.common import filesystem as _fs
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

_logger = utils.getLogger(__name__)

class CheckpointThread(threading.Thread):   #birbal added 
    def __init__(self, queue, checkpoint_dir, interval=60, batch_size=30):
        super().__init__()
        self.queue = queue
        self.checkpoint_dir = checkpoint_dir
        self.interval = interval
        self.batch_size = batch_size
        self._stop_event = threading.Event()
        self._buffer = []
        self._last_flush_time = time.time()

    def run(self):
        while not self._stop_event.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=1)
                self._buffer.append(item)
            except:
                pass  # No item fetched

            time_since_last_flush = time.time() - self._last_flush_time
            if len(self._buffer) >= self.batch_size or time_since_last_flush >= self.interval:
                self.flush_to_delta()
                self._buffer.clear()
                self._last_flush_time = time.time()

        # Final flush
        if self._buffer:
            self.flush_to_delta()

    def flush_to_delta(self):
        try:
            df = pd.DataFrame(self._buffer)
            if df.empty:
                _logger.info(f"[Checkpoint] DataFrame is empty. Skipping write to {self.checkpoint_dir}")
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.checkpoint_dir, f"checkpoint_{timestamp}.parquet")
            df.to_parquet(file_path, index=False)
            _logger.info(f"[Checkpoint] Saved {len(df)} records to {file_path}")
            
        except Exception as e:
            _logger.error(f"[Checkpoint] Failed to write to {self.checkpoint_dir}: {e}", exc_info=True)

    def stop(self):        
        self._stop_event.set()
        _logger.info("STOP event called..........")

    @staticmethod
    def load_processed_objects(checkpoint_dir, object_type= None):
        try:
            dataset = ds.dataset(checkpoint_dir, format="parquet")
            df = dataset.to_table().to_pandas()
            result_dict = {}

            if df.empty:
                _logger.warning(f"[Checkpoint] Parquet data is empty in {checkpoint_dir}")
                return {}

            if object_type == "experiments":
                result_dict = df.groupby("experiment_id")["run_id"].apply(lambda x: list(set(x))).to_dict()                              

            if object_type == "models":
                result_dict = df.groupby("model")["version"].apply(lambda x: list(set(x))).to_dict()   
                
            return result_dict

        except Exception as e:
            _logger.warning(f"[Checkpoint] Failed to load checkpoint data from {checkpoint_dir}: {e}", exc_info=True)
            return None

def filter_unprocessed_objects(checkpoint_dir,object_type,to_be_processed_objects):       #birbal added         
        processed_objects = CheckpointThread.load_processed_objects(checkpoint_dir,object_type)
        if isinstance(to_be_processed_objects, dict):   
            unprocessed_objects = {k: v for k, v in to_be_processed_objects.items() if k not in processed_objects}
            return unprocessed_objects, processed_objects
        
        if isinstance(to_be_processed_objects, list):   
            unprocessed_objects = list(set(to_be_processed_objects) - set(processed_objects.keys()))
            return unprocessed_objects, processed_objects
        
        return None,None
              
             