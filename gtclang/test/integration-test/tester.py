import sys
import os.path
import re
import subprocess
import filecmp
import argparse
from difflib import unified_diff


def print_error(message):
    print("FAILURE:", message, file=sys.stderr)


def get_line_number(content, m):
    return list(map(lambda x: m in x, content.split("\n"))).index(True) + 1


def run_test(content, gtclang_exec, filename, verbose=False):
    dirname, basename = (
        os.path.dirname(args.source),
        os.path.splitext(os.path.basename(args.source))[0],
    )
    cmd = []
    error_happened = False
    expect_error = False

    patterns = {
        "RUN": re.compile(r"//\s*RUN:\s*(?P<command>[^\n]+)"),
        "EXPECTED_LINE": re.compile(r"//\s*EXPECTED\s*%line%:\s*(?P<output>[^\n]+)"),
        "EXPECTED": re.compile(r"//\s*EXPECTED:\s*(?!%line%)(?P<output>[^\n]+)"),
        "EXPECTED_FILE": re.compile(
            r"//\s*EXPECTED_FILE:\s*OUTPUT:\s*(?P<output>[^\s]+)\s*REFERENCE:\s*(?P<reference>[^\s]+)"
        ),
        "EXPECTED_ERROR": re.compile(r"//\s*EXPECTED_ERROR:\s*(?P<output>[^\n]+)"),
    }

    # RUN
    m_runs = patterns["RUN"].findall(content)
    if len(m_runs) != 1 or m_runs[0] is None:
        raise ValueError("Requires exactly one RUN statement somewhere in the file!")
    else:
        cmd = (
            m_runs[0]
            .replace(r"%gtclang%", gtclang_exec)
            .replace(r"%file%", filename)
            .split(" ")
        )

    # Run it!
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if verbose:
        print(" ".join(cmd))
        print(proc.stdout.strip("\n"))

    # Begin tests

    # EXPECTED_LINE and EXPECTED
    m_expected = patterns["EXPECTED_LINE"].findall(content) + patterns[
        "EXPECTED"
    ].findall(content)
    for m in m_expected:
        # Replace all possible patterns with regex expressions
        m = m.strip(" ")
        line_match = re.search(r"%line[\+|\-]?\d*%", m)
        if line_match:
            line = eval(
                line_match.group()
                .strip("%")
                .replace("line", str(get_line_number(content, m)))
            )
            m = m.replace(line_match.group(), str(line))
        # Look for line in stdout
        if not re.search(m, proc.stdout):
            print_error("Could not match: {}".format(m))
            error_happened = True

    # EXPECTED_ERROR
    m_errors = patterns["EXPECTED_ERROR"].findall(content)
    for m in m_errors:
        # Replace all possible patterns with regex expressions
        m = m.strip(" ")
        # Look for line in stdout
        if re.search(m, proc.stderr) is None:
            print_error("Could not find error in stdout: {}".format(m))
            error_happened = True

    # If we expect an error, we want gtclang to return an error
    if len(m_errors) > 0:
        expect_error = True

    # EXPECTED_FILE
    m_expected_file = patterns["EXPECTED_FILE"].findall(content)
    for m in m_expected_file:
        # Replace %filename% in string expression and split on commas
        tests = zip(*(x.replace(r"%filename%", basename).split(",") for x in m))
        for output, reference in tests:
            # Add source directory path to reference
            reference = os.path.join(dirname, reference)

            if not os.path.exists(output):
                print_error("Could not find file: {}".format(output))
                error_happened = True
            if not os.path.exists(reference):
                print_error("Could not find file: {}".format(reference))
                error_happened = True

            # Compare files
            if not filecmp.cmp(output, reference, shallow=False):
                print_error("Files {} and {} do not match".format(output, reference))
                with open(output, mode="r") as f:
                    s1 = f.readlines()
                with open(reference, mode="r") as f:
                    s2 = f.readlines()
                sys.stdout.writelines(
                    unified_diff(s1, s2, fromfile=output, tofile=reference)
                )
                error_happened = True

    # Ensure the compiler was successful
    successful_code = 0
    # Boolean indicating whether gtclang returned the correct ret_val
    gtclang_success = (
        proc.returncode == successful_code
        if not expect_error
        else proc.returncode != successful_code
    )

    if not gtclang_success:
        print_error(
            "received return code {}{}".format(
                proc.returncode, ":\n{}".format(proc.stderr) if proc.stderr else ""
            )
        )
        return 1
    elif error_happened:
        return 1
    else:
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GTClang Source Code.")
    parser.add_argument("gtclang", type=str, help="GTClang executable")
    parser.add_argument("source", type=str, help="Source code to compile")
    parser.add_argument(
        "-v", "--verbose", help="modify output verbosity", action="store_true"
    )
    parser.add_argument("options", nargs="*")
    args = parser.parse_args()

    content = ""
    with open(args.source, mode="r") as f:
        content = f.read().rstrip("\n")
    ret_val = run_test(content, args.gtclang, args.source, verbose=args.verbose)
    sys.exit(ret_val)
