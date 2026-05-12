import importlib.util
from pathlib import Path


def main():
    script_path = Path(__file__).with_name("quest_tcp_aero_teleop.py")
    spec = importlib.util.spec_from_file_location("quest_tcp_aero_teleop", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()

