"""Event-based keyboard capture, including taps released between game decisions."""


class KeyboardCapture:
    def __init__(self, pygame):
        self.pg = pygame
        self.bindings = {pygame.K_LEFT: 1, pygame.K_a: 1, pygame.K_RIGHT: 2,
                         pygame.K_d: 2, pygame.K_SPACE: 3}
        self.held = set()
        self.pending = set()
        self.presses = [0, 0, 0]
        self.paused = True
        self.focused = True

    def clear(self):
        self.held.clear()
        self.pending.clear()
        self.presses = [0, 0, 0]

    def update(self, events):
        pg = self.pg
        for event in events:
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                raise KeyboardInterrupt
            if event.type in (pg.WINDOWFOCUSLOST, pg.WINDOWFOCUSGAINED):
                self.focused = event.type == pg.WINDOWFOCUSGAINED
                self.paused = True
                self.clear()
            elif event.type == pg.KEYDOWN and self.focused:
                if event.key == pg.K_RETURN and self.paused:
                    self.clear()
                    self.paused = False
                elif event.key in self.bindings:
                    if event.key not in self.held:
                        self.presses[self.bindings[event.key] - 1] += 1
                        self.pending.add(event.key)
                    self.held.add(event.key)
            elif event.type == pg.KEYUP:
                self.held.discard(event.key)

    def preview(self):
        keys = self.held if self.paused else self.held | self.pending
        choices = {self.bindings[key] for key in keys}
        # Preserve the documented one-button rule, including conflicting moves.
        return 3 if 3 in choices else 1 if choices == {1} else 2 if choices == {2} else 0

    def consume(self):
        if self.paused:
            raise ValueError("Cannot record input while paused")
        action = self.preview()
        held_actions = {self.bindings[key] for key in self.held}
        tap = action != 0 and action not in held_actions
        presses = self.presses.copy()
        self.pending.clear()
        self.presses = [0, 0, 0]
        return action, presses, tap
