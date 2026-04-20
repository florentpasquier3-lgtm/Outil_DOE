import streamlit.web.cli as stcli
import os, sys

def resolve_path(path):
    resolved_path = getattr(sys, '_MEIPASS', os.path.abspath(os.curdir))
    return os.path.join(resolved_path, path)

if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("app.py"),
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())