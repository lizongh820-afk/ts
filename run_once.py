import os
import subprocess
import sys

os.environ["GLM_API_KEY"] = "a7c440b626144a349993666356cdd74a.DpyYjq64DqJhtnJp"

script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "角色对话.py")

proc = subprocess.Popen(
    [sys.executable, script_path],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    errors="replace",
)

topic = "雨夜酒馆，几个陌生人被困在一起"
output, _ = proc.communicate(input=topic + "\n", timeout=600)

log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_output.log")
with open(log_path, "w", encoding="utf-8") as f:
    f.write(output)

print(output[-5000:] if len(output) > 5000 else output)
