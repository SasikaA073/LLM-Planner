# Code from https://github.com/askforalfred/alfred/blob/master/scripts/check_thor.py

from ai2thor.controller import Controller

print("Library loaded ")
c = Controller()
c.start()
event = c.step(dict(action="MoveAhead"))

print("Hello moved ahead")
assert event.frame.shape == (300, 300, 3)
print(event.frame.shape)
print("Everything works!!!")