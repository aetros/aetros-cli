import subprocess
import sys


def update_objects(ref, remote_url):
    commit_sha = subprocess.check_output(['git', 'rev-parse', ref]).decode('utf-8').strip()
    if not commit_sha:
        sys.stderr.write("Error: ref %s is not a commit\n" % (ref,))
        sys.exit(1)

    object_type = subprocess.check_output(['git', 'cat-file', '-t', ref]).decode('utf-8').strip()
    if 'commit' != object_type:
        sys.stderr.write("Error: ref %s is not a commit (but a %s)\n" % (ref, object_type))
        sys.exit(1)

    object_content = subprocess.check_output(['git', 'cat-file', '-p', ref]).decode('utf-8').strip()
    tree_sha = None
    for line in object_content.splitlines():
        if line.startswith('tree '):
            tree_sha = line[len('tree '):]

    if not tree_sha:
        sys.stderr.write("Error: Could not detect the tree for commit\n")
        sys.exit(1)

    object_content = subprocess.check_output(['git', 'ls-tree', '-r', tree_sha]).decode('utf-8').strip()
    object_shas = []

    for line in object_content.splitlines():
        exploded = line.split(' ')

        if len(exploded) != 3:
            sys.stderr.write("Error: Wrong line format of ls-tree for %s: %s\n" % (tree_sha, line, ))
            sys.exit(1)

        object_shas.append(str(exploded[2][:40]))

    shas_to_check = [commit_sha, tree_sha] + object_shas

    p = subprocess.Popen(
        ['ssh', remote_url, 'git-cat-file-check.sh', repo_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE
    )

    missing_object_shas, stderr = p.communicate('\n'.join(shas_to_check))

    if not missing_object_shas:
        sys.stdout.write("All objects are already known to the server.\n")
        sys.exit(0)

    # print(missing_object_shas)

    unpack_process = subprocess.Popen(['ssh', remote_url, 'git-unpack-objects.sh', repo_path, ref, commit_sha], stdin=subprocess.PIPE)
    pack_process = subprocess.Popen(['git', 'pack-objects', '--stdout', '--all-progress'],
                                    stdin=subprocess.PIPE, stdout=unpack_process.stdin)
    pack_process.communicate(missing_object_shas)

    pack_process.wait()
    unpack_process.wait()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('ref', help="Branch name or commit id")
    parser.add_argument('remote', nargs='?', help="Default origin")
    parsed_args = parser.parse_args(sys.argv[1:])

    ref = parsed_args.ref
    remote = parsed_args.remote if parsed_args.remote else 'origin'
    output = subprocess.check_output(['git', 'remote', 'get-url', '--push', remote]).decode('utf-8').strip()
    remote_url, repo_path = output.split(':')
    update_objects(ref, remote_url)
