import importlib
import pkgutil

from batchgenerators.utilities.file_and_folder_operations import *


def recursive_find_python_class(folder: str, class_name: str, current_module: str):
    # Most nnU-Net trainers use the class name as the module name. Importing
    # that exact module first avoids loading unrelated optional architectures.
    matching_module = join(folder, class_name + '.py')
    if isfile(matching_module):
        m = importlib.import_module(current_module + "." + class_name)
        if hasattr(m, class_name):
            return getattr(m, class_name)

    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules([folder]):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, class_name):
                tr = getattr(m, class_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules([folder]):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(join(folder, modname), class_name, current_module=next_current_module)
            if tr is not None:
                break
    return tr
