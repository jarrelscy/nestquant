#!/bin/bash
cd /home/coder/git/nestquant/threads/32-gbdt-sal
nice -n 10 /home/coder/git/glm52/.venv/bin/python leadlag.py > /tmp/nestquant/32-gbdt-sal/logs/leadlag.log 2>&1
