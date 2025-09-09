"""
(from cryodrgn)


Here we add modules under the `recovar.commands` 

See the `[project.scripts]` entry in the `pyproject.toml` file for how this module
is used to create the commands during installation. We list the modules to use
explicitly for each folder in case the namespace is inadvertantly polluted, and also
since automated scanning for command modules is computationally non-trivial.


"""
import os
import sys
import importlib
import argparse
from importlib import import_module
import re
import recovar

import os
import sys
import importlib

def main_commands() -> None:
    """Primary commands installed with recovar as `recovar <cmd_module_name>`."""
    cmd_dir = os.path.join(os.path.dirname(__file__), "commands")
    
    available_cmds = [
        os.path.splitext(filename)[0]
        for filename in os.listdir(cmd_dir)
        if filename.endswith(".py") and filename != "__init__.py"
    ]
    
    if len(sys.argv) < 2:
        print("Usage: recovar <command>")
        print("Available commands:")
        for cmd in available_cmds:
            print("  " + cmd)
        sys.exit(1)
    
    cmd_name = sys.argv[1]
    if cmd_name not in available_cmds:
        print(f"Command '{cmd_name}' not found.")
        print("Available commands:")
        for cmd in available_cmds:
            print("  " + cmd)
        sys.exit(1)
    
    # Remove the subcommand from sys.argv so that the subcommand's parser doesn't see it.
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    module_name = f"recovar.commands.{cmd_name}"
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        print(f"Error importing {module_name}: {e}")
        sys.exit(1)
    
    if hasattr(mod, "main"):
        mod.main()  # subcommand's main() will now see sys.argv without the subcommand name.
    else:
        print(f"Module {module_name} does not define a main() function.")
        sys.exit(1)

if __name__ == "__main__":
    main_commands()


# def _get_commands(cmd_dir: str, cmds: list[str], doc_str: str = "") -> None:
#     """Start up a command line interface using given modules as subcommands.

#     Arguments
#     ---------
#     cmd_dir:    path to folder containing recovar command modules
#     cmds:       list of commands in the above directory we want to use in the package
#     doc_str:    short documentation string describing this list of commands as a whole

#     """
#     parser = argparse.ArgumentParser(description=doc_str)
#     parser.add_argument(
#         "--version", action="version", version="recovar " + recovar.__version__
#     )

#     subparsers = parser.add_subparsers(title="Choose a command")
#     subparsers.required = True
#     dir_lbl = os.path.basename(cmd_dir)

#     # look for Python modules that have the `add_args` method defined, which is what we
#     # use to mark a module in these directories as added to the command namespace
#     for cmd in cmds:
#         module_name = ".".join(["recovar", dir_lbl, cmd])
#         module = import_module(module_name)

#         # if hasattr(module, "add_args"):
#         #     module.add_args(this_parser)

#         # if not hasattr(module, "add_args"):
#         #     raise RuntimeError(
#         #         f"Module `{cmd}` under `{cmd_dir}` does not have the required "
#         #         f"`add_args()` function defined; see other modules under the "
#         #         f"same directory for examples!"
#         #     )

#         # Parse the module-level documentation appearing at the top of the file
#         parsed_doc = module.__doc__.split("\n") if module.__doc__ else list()
#         descr_txt = parsed_doc[0] if parsed_doc else ""
#         epilog_txt = "" if len(parsed_doc) <= 1 else "\n".join(parsed_doc[1:])

#         # We have to manually re-add the backslashes used to break up lines
#         # for multi-line examples as these get parsed into spaces by .__doc__
#         # NOTE: This means command docstrings shouldn't otherwise have
#         # consecutive spaces!
#         epilog_txt = re.sub(" ([ ]+)", " \\\n\\1", epilog_txt)

#         # the docstring header becomes the help message "description", while
#         # the rest of the docstring becomes the "epilog"
#         this_parser = subparsers.add_parser(
#             cmd,
#             description=descr_txt,
#             epilog=epilog_txt,
#             formatter_class=argparse.RawTextHelpFormatter,
#         )
#         if hasattr(module, "add_args"):
#             module.add_args(this_parser)
#         # module.add_args(this_parser)
#         this_parser.set_defaults(func=module.main)

#     args = parser.parse_args()
#     args.func(args)


# def main_commands() -> None:
#     """Primary commands installed with recovar as `recovar <cmd_module_name>."""
#     _get_commands(
#         cmd_dir=os.path.join(os.path.dirname(__file__), "commands"),
#         cmds=[
#         "pipeline_with_outliers",
#         "pipeline",
#         "outlier_detection",
#         "extract_image_subset_from_kmeans",
#         "run_test_dataset",
#         "make_test_dataset",
#         # "make_spike_datasets",
#         "extract_image_subset",
#         "estimate_stable_states",
#         "compute_trajectory",
#         "estimate_conformational_density",
#         "compute_state",
#         # "compute_embedding",
#         "analyze"
#         ],
#         doc_str="Commands installed with recovar",
#     )

# def main_commands() -> None:
#     """Primary commands installed with recovar as `recovar <cmd_module_name>`."""
#     # Path to the commands directory
#     cmd_dir = os.path.join(os.path.dirname(__file__), "commands")
    
#     # List all Python files in the commands directory (ignoring __init__.py)
#     available_cmds = [
#         os.path.splitext(filename)[0]
#         for filename in os.listdir(cmd_dir)
#         if filename.endswith(".py") and filename != "__init__.py"
#     ]
    
#     if len(sys.argv) < 2:
#         print("Usage: recovar <command>")
#         print("Available commands:")
#         for cmd in available_cmds:
#             print("  " + cmd)
#         sys.exit(1)
    
#     cmd_name = sys.argv[1]
#     if cmd_name not in available_cmds:
#         print(f"Command '{cmd_name}' not found.")
#         print("Available commands:")
#         for cmd in available_cmds:
#             print("  " + cmd)
#         sys.exit(1)
    
#     # Dynamically import the command module from recovar.commands
#     module_name = f"recovar.commands.{cmd_name}"
#     try:
#         mod = importlib.import_module(module_name)
#     except ImportError as e:
#         print(f"Error importing {module_name}: {e}")
#         sys.exit(1)
    
#     # Remove the subcommand name so the module's own parser won't see it
#     sys.argv = [sys.argv[0]] + sys.argv[2:]
    
#     # Check for and execute the main() function in the module
#     if hasattr(mod, "main"):
#         mod.main()
#     else:
#         print(f"Module {module_name} does not define a main() function.")
#         sys.exit(1)

# # import argparse
# # import os
# # from importlib import import_module
# # import recovar

# import os
# import sys
# import importlib

# def main_commands() -> None:
#     """Primary commands installed with recovar as `recovar <cmd_module_name>`."""
#     # Path to the commands directory
#     cmd_dir = os.path.join(os.path.dirname(__file__), "commands")
    
#     # List all Python files in the commands directory (ignoring __init__.py)
#     available_cmds = [
#         os.path.splitext(filename)[0]
#         for filename in os.listdir(cmd_dir)
#         if filename.endswith(".py") and filename != "__init__.py"
#     ]
    
#     if len(sys.argv) < 2:
#         print("Usage: recovar <command>")
#         print("Available commands:")
#         for cmd in available_cmds:
#             print("  " + cmd)
#         sys.exit(1)
    
#     cmd_name = sys.argv[1]
#     if cmd_name not in available_cmds:
#         print(f"Command '{cmd_name}' not found.")
#         print("Available commands:")
#         for cmd in available_cmds:
#             print("  " + cmd)
#         sys.exit(1)
    
#     # Dynamically import the command module from recovar.commands
#     module_name = f"recovar.commands.{cmd_name}"
#     try:
#         mod = importlib.import_module(module_name)
#     except ImportError as e:
#         print(f"Error importing {module_name}: {e}")
#         sys.exit(1)
    
#     # Check for and execute the main() function in the module
#     if hasattr(mod, "main"):
#         mod.main()
#     else:
#         print(f"Module {module_name} does not define a main() function.")
#         sys.exit(1)

