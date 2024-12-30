import sys
import pickle
import EBD_SP_Cut_Empty_unlimited_threads
import EBD_SP_Cut_Empty_unlimited_threads_SP_only
import EBD_SP_Cut_unlimited_threads_new_sep_algo

pickle_folder=sys.argv[1]
task_id=int(sys.argv[2])

print(f"Pickle file {pickle_folder}")
print(f"task ID {task_id}")

with open(pickle_folder,"rb") as f:
    task=pickle.load(f)[task_id]

model_name=task[0]
if model_name=="EBD_SP_Cut_Empty":
    EBD_SP_Cut_unlimited_threads_new_sep_algo.execute_task(task)



