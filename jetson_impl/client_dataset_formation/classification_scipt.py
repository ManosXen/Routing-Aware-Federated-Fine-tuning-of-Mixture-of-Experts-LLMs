import os

task_list = [
    ["allenai/winogrande", "train", "sentence"],
    #["baber/piqa", "train", "goal"],
    #["tau/commonsense_qa", "train", "question_concept+question"],
    ["jet-ai/social_i_qa", "train", "context+question"],
    #["aps/super_glue", "train", "passage+question"],
    #["ehovy/race", "train", "article+question"],
]

for task in task_list:
    if "winogrande" in task[0] or "social" in task[0] or "race" in task[0]:
        command = (
            "docker run -d --rm --gpus 'device=4' "
            "-e OPENBLAS_NUM_THREADS=64 "
            "-v $(pwd):/files/ "
            f"bnb_transformers python3 /files/impl_v2/client_dataset_formation/dataset_clustering.py {task[0]} {task[1]} {task[2]}"
        )
    else:
        command = (
            "docker run -d --rm --gpus 'device=5' "
            "-v $(pwd):/files/ "
            f"bnb_transformers python3 /files/impl_v2/client_dataset_formation/dataset_clustering.py {task[0]} {task[1]} {task[2]}"
        )    
    print(f"Running command: {command}", flush=True)
    os.system(command)