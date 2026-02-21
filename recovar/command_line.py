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
