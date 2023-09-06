import os

lrs = [
    '2e-5',
    '1e-5',
    '4e-5'
]

for lr in lrs:
    lines = open('run.sh').readlines()
    lines[0] = 'LR={}\n'.format(lr)
    f = open('run.sh', 'w')
    for line in lines:
        f.write(line)
    f.close()
    os.system('sh run.sh')