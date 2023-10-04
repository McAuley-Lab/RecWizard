import sys, os
import shutil
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from recwizard.utility import HF_ORG
sys.path.append('../')
sys.path.append('../src/recwizard')


HF_ORG = "recwizard"


def copy_from_source_to_destination(source_folder: str, destination_folder: str):
    for filename in os.listdir(source_folder):
        source = os.path.join(source_folder, filename)
        destination = os.path.join(destination_folder, filename)

        # Copy each file to the destination folder
        shutil.copy(source, destination)

if __name__ == '__main__':
    dataset_path = sys.argv[1]
    dataset = load_dataset(dataset_path, download_mode='force_redownload')
    repo_id = f"{HF_ORG}/{dataset_path}"
    # dataset.push_to_hub(repo_id)
    local_repo = f"tmp/{repo_id}"
    repo_url = create_repo(repo_id, repo_type="dataset", exist_ok=True)
    if os.path.exists(local_repo):
        clone_from = None
    else:
        clone_from = repo_url
    repo = Repository(
        local_dir=local_repo,
        clone_from=clone_from,
        skip_lfs_files=True
    )

    repo.git_pull()
    copy_from_source_to_destination(dataset_path, local_repo)
    repo.git_add('*.py')
    repo.git_add('*.json')
    repo.git_add('*.jsonl')
    repo.git_commit(input("Please enter the commit message:"))
    repo.git_push()

