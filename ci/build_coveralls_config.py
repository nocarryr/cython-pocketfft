import os
import subprocess
import shlex
import yaml

IS_PR = os.environ.get('IS_PR_BUILD')
PR_SOURCE_COMMIT = os.environ.get('PR_SOURCE_COMMIT')

def run_command(cmd_str):
    result = subprocess.check_output(shlex.split(cmd_str))
    if isinstance(result, bytes):
        result = result.decode('UTF-8')
    return result

def gitlog(fmt, commit='-1'):
    cmd_str = 'git --no-pager log {commit} --pretty=format:{fmt}'.format(
        commit=commit, fmt=fmt,
    )
    return run_command(cmd_str)

def get_env_vars():
    keys = ['service_name', 'service_number', 'service_job_id']
    if IS_PR:
        keys.append('service_pull_request')
    envkeys = {key:'_'.join(['COVERALLS', key.upper()]) for key in keys}
    conf = {key:os.environ[envkey] for key, envkey in envkeys.items()}
    conf['parallel'] = True
    return conf


def get_git_info():
    if not IS_PR:
        return {}
    merge_commit = '-1'
    branch = os.environ['GIT_BRANCH']
    head = {
        'id': gitlog('%H', merge_commit),
        'author_name': gitlog('%aN', PR_SOURCE_COMMIT),
        'author_email': gitlog('%ae', PR_SOURCE_COMMIT),
        'committer_name': gitlog('%cN', merge_commit),
        'committer_email': gitlog('%ce', merge_commit),
        'message': gitlog('%s', PR_SOURCE_COMMIT),
    }
    remotes = [{'name': line.split()[0], 'url': line.split()[1]}
               for line in run_command('git remote -v').splitlines()
               if '(fetch)' in line]
    return {
        'git': {
            'branch': branch,
            'head': head,
            'remotes': remotes,
        },
    }


conf = get_env_vars()
conf.update(get_git_info())
print('coveralls config: {}'.format(conf))
with open('.coveralls.yml', 'w') as f:
    f.write(yaml.dump(conf, Dumper=yaml.Dumper))
