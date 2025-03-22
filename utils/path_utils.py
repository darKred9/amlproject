import os

# get specified path from the root directory of this project
def get_path_from_project_root(*path_components):
    project_root = os.environ.get('PROJECT_ROOT')
    
    if not project_root:
        current_dir = os.getcwd()
        project_root = current_dir
        
        while current_dir and not os.path.exists(os.path.join(current_dir, '.git')):
            parent = os.path.dirname(current_dir)
            if parent == current_dir:
                break
            current_dir = parent
            if os.path.exists(os.path.join(current_dir, '.git')):
                project_root = current_dir
    
    return os.path.join(project_root, *path_components)


if __name__ == "__main__":
    # use case
    example_path = get_path_from_project_root("data", "raw", "example.csv")
    print(f"path: {example_path}")