"""
Run embedded integration tests from GTClang DSL code.
"""
import sys
import os.path
import re
import subprocess
import filecmp
import argparse
from json import loads as load_json
from difflib import unified_diff

patterns = {
    "RUN": re.compile(r"//\s*RUN:\s*(?P<command>[^\n]+)"),
    "EXPECTED_LINE": re.compile(r"//\s*EXPECTED\s*%line%:\s*(?P<output>[^\n]+)"),
    "EXPECTED": re.compile(r"//\s*EXPECTED:\s*(?!%line%)(?P<output>[^\n]+)"),
    "EXPECTED_FILE": re.compile(
        r"//\s*EXPECTED_FILE:\s*OUTPUT:\s*(?P<output>[^\s]+)\s*REFERENCE:\s*(?P<reference>[^\s]+)(?P<remainder>[^\n]*)"
    ),
    "EXPECTED_ERROR": re.compile(r"//\s*EXPECTED_ERROR:\s*(?P<output>[^\n]+)"),
}


def print_error(message):
    print("FAILURE: " + message, file=sys.stderr)


def print_test(message):
    print("TEST:", message)


def compare_json_files(output, reference, ignore_keys=[]):
    """Compare JSON files, ignore certain keys"""

    def read_file(filename):
        if not os.path.exists(filename):
            raise ValueError("Could not find file: {}".format(filename))
        else:
            with open(filename, mode="r") as f:
                content = f.readlines()
        return content

    def compare_json_trees(t1, t2):
        # If entering here, t1 and t2 could be lists or dicts (of values, lists, or dicts)
        if isinstance(t1, list):
            for v1, v2 in zip(t1, t2):
                # Then v1 and v2 are _values_
                if any(isinstance(v1, x) for x in (list, dict)):
                    if not compare_json_trees(v1, v2):
                        return False
                if v1 != v2:
                    msg = "Values " + str(v1) + " and " + str(v2) + " do not match"
                    print_error(msg)
                    return False
        elif isinstance(t1, dict):
            for v1, v2 in zip(set(t1), set(t2)):
                # Then v1 and v2 are _keys_
                if any(isinstance(t1[v1], x) for x in (list, dict)):
                    if not compare_json_trees(t1[v1], t2[v2]):
                        return False
                if v1 != v2 and v1 not in ignore_keys:
                    msg = "Values " + str(v1) + " and " + str(v2) + " do not match"
                    print_error(msg)
                    return False
                if t1[v1] != t2[v2] and v1 not in ignore_keys:
                    msg = (
                        "Values "
                        + str(t1[v1])
                        + " and "
                        + str(t2[v2])
                        + " do not match"
                    )
                    print_error(msg)
                    return False
        else:
            raise ValueError("Logic error")
        return True

    output_lines = read_file(output)
    output_json = load_json(" ".join(output_lines))
    reference_lines = read_file(reference)
    reference_json = load_json(" ".join(reference_lines))

    if not compare_json_trees(reference_json, output_json):
        # Compare files
        sys.stdout.writelines(
            unified_diff(
                [l for l in output_lines if not any((p in l for p in ignore_keys))],
                [l for l in reference_lines if not any((p in l for p in ignore_keys))],
                fromfile=output,
                tofile=reference,
            )
        )
        return False
    return True


def run_test(content, gtclang_exec, filename, verbose=False, ignore_keys=[]):
    """Main test function."""

    def get_line_number(content, m):
        return list(map(lambda x: m in x, content.split("\n"))).index(True) + 1

    dirname, basename = (
        os.path.dirname(args.source),
        os.path.splitext(os.path.basename(args.source))[0],
    )
    cmd = []
    error_happened = False
    expect_error = False

    # Look for RUN
    m_runs = patterns["RUN"].findall(content)
    if len(m_runs) != 1 or m_runs[0] is None:
        raise ValueError("Requires exactly one RUN statement somewhere in the file!")
    else:
        cmd = (
            m_runs[0]
            .replace(r"%gtclang%", gtclang_exec)
            .replace(r"%file%", filename)
            .replace(r"%filename%", basename)
            .split(" ")
        )

    # Run it!
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = (x.decode() for x in (proc.stdout, proc.stderr))
    print(" ".join(cmd))
    if verbose:
        print(stdout.strip("\n"))
        print(stderr.strip("\n"))

    # Begin tests

    # Look for EXPECTED_LINE and EXPECTED
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
        print_test("EXPECTED_LINE: " + m)
        # Look for line in stdout
        if not re.search(m, stdout):
            print_error("Could not match: {}".format(m))
            error_happened = True

    # Look for EXPECTED_ERROR
    m_errors = patterns["EXPECTED_ERROR"].findall(content)
    for m in m_errors:
        # Replace all possible patterns with regex expressions
        m = m.strip(" ")
        print_test("EXPECTED_ERROR: " + m)
        # Look for line in stdout
        if re.search(m, stderr) is None:
            print_error("Could not find error in stdout: {}".format(m))
            error_happened = True

    # If we expect an error, we want gtclang to return an error
    if len(m_errors) > 0:
        expect_error = True

    # Look for EXPECTED_FILE
    m_expected_file = patterns["EXPECTED_FILE"].findall(content)
    for m in m_expected_file:
        files = (m[0], m[1])
        if len(m) > 2:
            # test for IGNORE pattern
            ignore_keys += re.findall(r"IGNORE:\s*(?P<pattern>[^ ]+)", m[2])
        # Replace %filename% in string expression and split on commas
        tests = zip(*(x.replace(r"%filename%", basename).split(",") for x in files))
        for output, reference in tests:
            # Add source directory path to reference
            reference = os.path.join(dirname, reference)
            print_test(
                "EXPECTED_FILE: OUTPUT={} REFERENCE={}{}".format(
                    output, reference, "".join(" IGNORE: " + k for k in ignore_keys)
                )
            )

            if all(
                (
                    os.path.splitext(f)[-1] in (".iir", ".sir", ".json")
                    for f in (output, reference)
                )
            ):
                # All these are json trees, so we can compare them
                same_trees = compare_json_files(
                    output, reference, ignore_keys=ignore_keys
                )
                if not same_trees:
                    error_happened = True
                    print_error("JSON trees do not match")

            else:
                raise ValueError("Not yet implemented.")

    # Ensure the compiler was successful
    successful_code = 0
    # Boolean indicating whether gtclang returned the correct ret_val
    gtclang_success = (
        proc.returncode == successful_code
        if not expect_error
        else proc.returncode != successful_code
    )

    # Return
    if not gtclang_success:
        print_error(
            "received return code {}{}".format(
                proc.returncode, ":\n{}".format(stderr) if stderr else "",
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
        "-i",
        "--ignore",
        nargs="*",
        help="Extra keys to ignore in comparison",
        default=[],
    )
    parser.add_argument(
        "-v", "--verbose", help="modify output verbosity", action="store_true"
    )
    parser.add_argument("options", nargs="*")
    args = parser.parse_args()

    # Read file
    with open(args.source, mode="r") as f:
        content = f.read().rstrip("\n")

    # Call test function
    ret_val = run_test(
        content,
        args.gtclang,
        args.source,
        verbose=args.verbose,
        ignore_keys=args.ignore,
    )
    sys.exit(ret_val)
