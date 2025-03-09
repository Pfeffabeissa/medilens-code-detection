import os
import subprocess
import argparse

parser = argparse.ArgumentParser(description="Install dependencies, with optional no-cache mode.")
parser.add_argument("--no-cache", action="store_true", help="Use --no-cache-dir for smaller Docker images")
args = parser.parse_args()


try:
    import dotenv
except ImportError:
    print("🔄 Installing required package: python-dotenv...\n")
    subprocess.run(["pip", "install", "python-dotenv"], check=True)
    import dotenv

from dotenv import load_dotenv

load_dotenv()
token = os.getenv("GH_PAT")

if not token:
    raise ValueError("❌ ERROR: GH_PAT is not set. Please check your .env file!")


public_deps = []
private_deps = []

try:
    with open("requirements.txt", "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):  # Ignore empty lines & comments
                continue
            if "TOKEN_PLACEHOLDER" in line:
                private_deps.append(line)
            else:
                public_deps.append(line)
except UnicodeDecodeError:
    print("❌ ERROR: requirements.txt is not in UTF-8 format. Please re-save it as UTF-8.")
    exit(1)

pip_install_cmd = ["pip", "install", "-U"]
if args.no_cache:
    pip_install_cmd.append("--no-cache-dir")
    print("🔄 Installing dependencies with --no-cache-dir...\n")

if public_deps:
    print("🔄 Installing public dependencies...\n")
    subprocess.run(pip_install_cmd + public_deps, check=True)

for private_dep in private_deps:
    print(f"🔄 Installing private dependency: {private_dep}\n")
    private_dep = private_dep.replace("TOKEN_PLACEHOLDER", token)
    subprocess.run(pip_install_cmd + ["--force-reinstall", private_dep], check=True)

print("✅ All dependencies installed successfully!")