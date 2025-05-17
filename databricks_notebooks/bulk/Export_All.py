# Databricks notebook source
# MAGIC %md ## Export All
# MAGIC
# MAGIC Export all the MLflow registered models and all experiments of a tracking server.
# MAGIC
# MAGIC **Widgets**
# MAGIC * `1. Output directory` - shared directory between source and destination workspaces.
# MAGIC * `2. Stages` - comma seperated stages to be exported.
# MAGIC * `3. Export latest versions` - export all or just the "latest" versions.
# MAGIC * `4. Run start date` - Export runs after this UTC date (inclusive). Example: `2023-04-05`.
# MAGIC * `5. Export permissions` - export Databricks permissions.
# MAGIC * `6. Export deleted runs`
# MAGIC * `7. Export version MLflow model`
# MAGIC * `8. Notebook formats`
# MAGIC * `9. Use threads`

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

from mlflow_export_import.bulk import config
from datetime import datetime
import time

# COMMAND ----------

output_dir = dbutils.widgets.get("output_dir")
output_dir = output_dir.replace("dbfs:","/dbfs")

stages = dbutils.widgets.get("stages")

export_latest_versions = dbutils.widgets.get("export_latest_versions") == "true"

run_start_date = dbutils.widgets.get("run_start_date")

export_permissions = dbutils.widgets.get("export_permissions") == "true"

export_deleted_runs = dbutils.widgets.get("export_deleted_runs") == "true"

export_version_model = dbutils.widgets.get("export_version_model") == "true"

notebook_formats = dbutils.widgets.get("notebook_formats").split(",")

use_threads = dbutils.widgets.get("use_threads") == "true"

task_index = int(dbutils.widgets.get("task_index"))

num_tasks = int(dbutils.widgets.get("num_tasks"))

run_timestamp = int(dbutils.widgets.get("run_timestamp"))
 
if run_start_date=="": run_start_date = None

print("output_dir:", output_dir)
print("stages:", stages)
print("export_latest_versions:", export_latest_versions)
print("run_start_date:", run_start_date)
print("export_permissions:", export_permissions)
print("export_deleted_runs:", export_deleted_runs)
print("export_version_model:", export_version_model)
print("notebook_formats:", notebook_formats)
print("use_threads:", use_threads)
print("task_index:", task_index)
print("num_tasks:", num_tasks)
print("run_timestamp:", run_timestamp)

# COMMAND ----------

assert_widget(output_dir, "1. Output directory")

# COMMAND ----------

output_dir = f"{output_dir}/{run_timestamp}/{task_index}"
output_dir

# COMMAND ----------

log_path=f"/tmp/my.log"
log_path

# COMMAND ----------

config.log_path=log_path

# COMMAND ----------

from mlflow_export_import.bulk.export_all import export_all

export_all(
    output_dir = output_dir, 
    stages = stages,
    export_latest_versions = export_latest_versions,
    run_start_time = run_start_date,
    export_permissions = export_permissions,
    export_deleted_runs = export_deleted_runs,
    export_version_model = export_version_model,
    notebook_formats = notebook_formats, 
    use_threads = use_threads,
    task_index = task_index,
    num_tasks = num_tasks
)

# COMMAND ----------

time.sleep(10)

# COMMAND ----------

# MAGIC %sh cat /tmp/my.log

# COMMAND ----------

dbfs_log_path = f"{output_dir}/export_all.log"
dbfs_log_path = dbfs_log_path.replace("/dbfs","dbfs:")
dbfs_log_path

# COMMAND ----------


dbutils.fs.cp(f"file:{log_path}", dbfs_log_path)

# COMMAND ----------

print(dbutils.fs.head(dbfs_log_path))
