"""CLI contract checks; no GPU or graph inputs required."""
import subprocess
import sys


def main():
    executable = sys.argv[1]
    checks = [
        (["--help"], 0, "--repair-max-rounds"),
        (["--repair-max-rounds", "1"], 1, "Repair limits require --repair"),
        (["--repair-max-rounds", "-1"], 1, "Invalid repair limit"),
        (["--repair-payload-mib", "0"], 1, "Invalid repair limit"),
        (["--repair", "--cpu", "--toy"], 1, "--repair currently requires"),
        (["--repair", "--gpu", "--source", "unused", "--target", "unused"],
         1, "--repair currently requires"),
        (["--repair", "--cpu", "--source", "unused", "--target", "unused",
          "--candidate-backend", "structural-index"], 1, "--repair currently requires"),
    ]
    for arguments, code, message in checks:
        result = subprocess.run([executable, *arguments], capture_output=True, text=True)
        if result.returncode != code or message not in result.stdout + result.stderr:
            raise AssertionError((arguments, result.returncode, result.stdout, result.stderr))
    print("Candidate repair CLI checks passed")


if __name__ == "__main__":
    main()
