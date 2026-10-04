import os
import sys


def resource_path(relative_path: str) -> str:
    if getattr(sys, "frozen", False):
        base_path = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


def suppress_first_run_prompt():
    from pathlib import Path

    config_dir = Path.home() / ".streamlit"
    config_dir.mkdir(parents=True, exist_ok=True)
    credentials_file = config_dir / "credentials.toml"
    if not credentials_file.exists():
        credentials_file.write_text('[general]\nemail = ""\n')


if __name__ == "__main__":
    suppress_first_run_prompt()

    from streamlit.web import cli as stcli

    app_path = resource_path("app.py")

    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
    ]
    sys.exit(stcli.main())
