import sys
sys.argv = sys.argv[:3]
src = open('compose.py').read()
head, tail = src.split('# skip-refinement variant')
pre = head.split('CASES =')[0]
exec(pre + '\n# skip-refinement variant' + tail)
