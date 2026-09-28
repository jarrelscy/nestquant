import sys
src = open('compose.py').read().split('# skip-refinement variant')[0]
exec(src)
