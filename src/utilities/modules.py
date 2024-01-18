import os
import sys
import importlib
import pkgutil
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

def recursive_find_python_class(name, folder=None, current_module="src.models", exit_if_not_found=True):
    # Set default search path to root modules
    project_root = get_project_root()

    # Construct the folder path from the project root
    if folder is None:
        folder_path = project_root / Path(*current_module.split("."))
        folder = [str(folder_path)]

    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules(folder):
        if not ispkg:
            module_name = f"{current_module}.{modname}"
            m = importlib.import_module(module_name)
            if hasattr(m, name):
                tr = getattr(m, name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = f"{current_module}.{modname}"
                print(next_current_module)
                tr = recursive_find_python_class(
                    name,
                    folder=[os.path.join(folder[0], modname)],
                    current_module=next_current_module,
                    exit_if_not_found=exit_if_not_found,
                )

            if tr is not None:
                break

    if tr is None and exit_if_not_found:
        sys.exit(f"Could not find module {name}")

    return tr