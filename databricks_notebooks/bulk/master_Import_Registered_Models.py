# Databricks notebook source
import requests
import json
import os

# COMMAND ----------

dbutils.widgets.text("1. Input directory", "") 
input_dir = dbutils.widgets.get("1. Input directory")
input_dir = input_dir.replace("dbfs:","/dbfs")

dbutils.widgets.dropdown("2. Delete model","no",["yes","no"])
delete_model = dbutils.widgets.get("2. Delete model") == "yes"

dbutils.widgets.text("3. Model rename file","")
val = dbutils.widgets.get("3. Model rename file") 
model_rename_file = val or None 

dbutils.widgets.text("4. Experiment rename file","")
val = dbutils.widgets.get("4. Experiment rename file") 
experiment_rename_file = val or None 

dbutils.widgets.dropdown("5. Import permissions","no",["yes","no"])
import_permissions = dbutils.widgets.get("5. Import permissions") == "yes"

dbutils.widgets.dropdown("6. Import source tags","no",["yes","no"])
import_source_tags = dbutils.widgets.get("6. Import source tags") == "yes"

dbutils.widgets.dropdown("7. Use threads","no",["yes","no"])
use_threads = dbutils.widgets.get("7. Use threads") == "yes"

dbutils.widgets.text("8. num_tasks", "") 
num_tasks = dbutils.widgets.get("8. num_tasks")

dbutils.widgets.text("9. log_directory", "") 
log_directory = dbutils.widgets.get("9. log_directory")

print("input_dir:", input_dir)
print("delete_model:", delete_model)
print("model_rename_file:", model_rename_file)
print("experiment_rename_file:", experiment_rename_file)
print("import_permissions:", import_permissions)
print("import_source_tags:", import_source_tags)
print("use_threads:", use_threads)
print("num_tasks:", num_tasks)
print("log_directory:", log_directory)

# COMMAND ----------

if input_dir.startswith("/Workspace"):
    input_dir=input_dir.replace("/Workspace","file:/Workspace") 

input_dir

# COMMAND ----------

DATABRICKS_INSTANCE=dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get('browserHostName').getOrElse(None)
DATABRICKS_INSTANCE = f"https://{DATABRICKS_INSTANCE}"
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

driver_node_type = "Standard_D4ds_v5"
worker_node_type = "Standard_D4ds_v5"

def create_multi_task_job_json(input_dir, delete_model, model_rename_file, experiment_rename_file, import_permissions, import_source_tags, use_threads, num_tasks):
    tasks = []
    for i in range(1, int(num_tasks)+1):
        task = {
            "task_key": f"task_{i}",
            "description": f"Bir Task for param1 = {i}",
            "new_cluster": {
                "spark_version": "15.4.x-cpu-ml-scala2.12",
                "node_type_id": worker_node_type,
                "driver_node_type_id": driver_node_type,
                "num_workers": 1,
                "data_security_mode": "SINGLE_USER",
                "runtime_engine": "STANDARD"
            },
            "notebook_task": {
                "notebook_path": "/Workspace/Users/birbal.das@databricks.com/mlflow/bir-mlflow-export-import/databricks_notebooks/bulk/Import_Registered_Models",
                "base_parameters": {
                    "input_dir": os.path.join(input_dir,str(i)),
                    "delete_model": delete_model,
                    "model_rename_file": model_rename_file,
                    "experiment_rename_file": experiment_rename_file,
                    "import_permissions": import_permissions,
                    "import_source_tags": import_source_tags,
                    "use_threads": use_threads,
                    "log_directory": os.path.join(log_directory,"{{job.start_time.iso_date}}-Import-jobid-{{job.id}}-jobrunid-{{job.run_id}}",str(i)),
                    "task_index": str(i)
                }
            }
        }
        tasks.append(task)

    job_json = {
        "name": "Import_Registered_Models_job",
        "tasks": tasks,
        "format": "MULTI_TASK"
    }

    return job_json

def submit_databricks_job():
    job_payload = create_multi_task_job_json(input_dir, delete_model, model_rename_file, experiment_rename_file, import_permissions, import_source_tags, use_threads, num_tasks)

    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        f"{DATABRICKS_INSTANCE}/api/2.2/jobs/create",
        headers=headers,
        data=json.dumps(job_payload)
    )

    if response.status_code == 200:
        print("Job submitted successfully.")
        print("Response:", response.json())
    else:
        print("Error submitting job:", response.status_code, response.text)



# COMMAND ----------

submit_databricks_job()
