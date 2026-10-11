"""Original pixel-inspired circuit art, bounded effects and geometry-derived maps."""
from collections import OrderedDict, deque
import math
import random
import pygame

INK = (13, 24, 27)
PANEL = (19, 33, 36)
WHITE = (235, 241, 225)
MUTED = (143, 166, 161)
LIME = (205, 247, 92)
RED = (255, 111, 94)


class Typography:
    def __init__(self):
        self.fonts = {}

    def text(self, surface, text, position, size=20, color=WHITE, bold=False, align="left", mono=False):
        key = (size, bold, mono)
        if key not in self.fonts:
            self.fonts[key] = pygame.font.SysFont("consolas" if mono else "bahnschrift", size, bold=bold)
        image = self.fonts[key].render(str(text), True, color)
        rect = image.get_rect()
        setattr(rect, {"left": "topleft", "center": "midtop", "right": "topright"}[align], position)
        surface.blit(image, rect)
        return rect


def format_time(seconds):
    if seconds is None:
        return "--:--.---"
    milliseconds = max(0, round(seconds*1000))
    return f"{milliseconds//60000:02}:{milliseconds//1000%60:02}.{milliseconds%1000:03}"


class CircuitArt:
    CHUNK_METERS = 64
    BASE_PPM = 5

    def __init__(self, track):
        self.track = track
        self.chunks = OrderedDict()
        self.decorations = {}
        self.grass = {"meadow": (52, 100, 65), "valley": (49, 88, 70), "coastal": (68, 109, 90)}[track.theme]
        self.random = random.Random(int(track.version[:8], 16))
        x0, y0, x1, y1 = track.bounds
        for _ in range(1600):
            x, y = self.random.uniform(x0-100, x1+100), self.random.uniform(y0-100, y1+100)
            p = track.project(x, y)
            if p.distance > p.width/2+track.RUNOFF_WIDTH+5:
                cell = (math.floor(x/self.CHUNK_METERS), math.floor(y/self.CHUNK_METERS))
                self.decorations.setdefault(cell, []).append((x, y, self.random.randrange(3)))

    def _chunk(self, gx, gy, quality):
        key = (gx, gy, quality)
        if key in self.chunks:
            self.chunks.move_to_end(key)
            return self.chunks[key]
        size = self.CHUNK_METERS*self.BASE_PPM
        surface = pygame.Surface((size, size))
        surface.fill(self.grass)
        rng = random.Random(gx*928371+gy*192837+int(self.track.version[:8], 16))
        for _ in range(90 if quality == "high" else 25):
            x, y = rng.randrange(size), rng.randrange(size)
            color = tuple(max(0, channel+rng.choice((-8, 8))) for channel in self.grass)
            pygame.draw.rect(surface, color, (x, y, rng.randrange(2, 7), 2))
        ox, oy = gx*self.CHUNK_METERS, gy*self.CHUNK_METERS
        def point(p):
            return round((p[0]-ox)*self.BASE_PPM), round((p[1]-oy)*self.BASE_PPM)
        n = len(self.track.points)
        visible = []
        for i in range(n):
            x, y = self.track.points[i]
            if ox-30 <= x <= ox+self.CHUNK_METERS+30 and oy-30 <= y <= oy+self.CHUNK_METERS+30:
                visible.append(i)
        for extra, color in ((7.0, (81, 101, 77)), (self.track.CURB_WIDTH, (227, 227, 207)), (0, (46, 54, 57))):
            for i in visible:
                j = (i+1)%n
                a, b = self.track.points[i], self.track.points[j]
                ni, nj = self.track.normals[i], self.track.normals[j]
                wi, wj = self.track.widths[i]/2+extra, self.track.widths[j]/2+extra
                pygame.draw.polygon(surface, color, [point(a-ni*wi), point(b-nj*wj), point(b+nj*wj), point(a+ni*wi)])
        for i in visible:
            j = (i+1)%n
            a, b = self.track.points[i], self.track.points[j]
            ni, nj = self.track.normals[i], self.track.normals[j]
            wi, wj = self.track.widths[i]/2, self.track.widths[j]/2
            curb_color = (207, 70, 66) if int(self.track.cumulative[i]/5)%2 else (231, 231, 212)
            for sign in (-1, 1):
                pygame.draw.polygon(surface, curb_color,
                                    [point(a+ni*sign*wi), point(b+nj*sign*wj),
                                     point(b+nj*sign*(wj+self.track.CURB_WIDTH)), point(a+ni*sign*(wi+self.track.CURB_WIDTH))])
                pygame.draw.line(surface, (194, 203, 190), point(a+ni*sign*(wi+self.track.RUNOFF_WIDTH)), point(b+nj*sign*(wj+self.track.RUNOFF_WIDTH)), 3)
                if int(self.track.cumulative[i]/8)%3 == 0:
                    pygame.draw.line(surface, (102, 125, 120), point(a+ni*sign*(wi+self.track.RUNOFF_WIDTH)), point(b+nj*sign*(wj+self.track.RUNOFF_WIDTH)), 2)
            if quality == "high" and i%3 == 0:
                pygame.draw.line(surface, (52, 60, 62), point(a+ni*wi*.15), point(b+nj*wj*.15), 1)
        # Checkered finish and grid boxes are aligned with the circuit tangent.
        finish = self.track.finish
        tx, ty = finish.tangent
        nx, ny = -ty, tx
        for row in range(2):
            for cell in range(math.ceil(finish.half_width*2/1.3)):
                lateral = -finish.half_width+cell*1.3
                corners = []
                for u, v in ((row*.9, lateral), ((row+1)*.9, lateral), ((row+1)*.9, lateral+1.3), (row*.9, lateral+1.3)):
                    corners.append(point((finish.center[0]+tx*u+nx*v, finish.center[1]+ty*u+ny*v)))
                pygame.draw.polygon(surface, WHITE if (cell+row)%2 else INK, corners)
        for back in (8, 18, 28, 38):
            x, y, angle = self.track.pose(self.track.start_progress-back)
            t, normal = (math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle))
            for side in (-1, 1):
                line = [(x+t[0]*u+normal[0]*v, y+t[1]*u+normal[1]*v) for u, v in ((-2, side*3-1), (2, side*3-1), (2, side*3+1), (-2, side*3+1))]
                pygame.draw.lines(surface, (143, 152, 142), False, [point(p) for p in line], 1)
        decorations = [item for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                       for item in self.decorations.get((gx+dx, gy+dy), ())]
        for x, y, kind in decorations:
            px, py = point((x, y))
            radius = (11, 15, 18)[kind]
            if px+radius*2 < 0 or py+radius*2 < 0 or px-radius > size or py-radius > size:
                continue
            pygame.draw.ellipse(surface, (35, 64, 48), (px-4, py+6, radius*2, radius))
            pygame.draw.rect(surface, (96, 78, 57), (px-2, py+3, 4, 11))
            pygame.draw.circle(surface, (31, 70, 54), (px, py), radius)
            pygame.draw.circle(surface, (56, 123, 76), (px-3, py-4), radius-3)
            pygame.draw.rect(surface, (85, 143, 84), (px-radius//2, py-radius//2, radius//2, 3))
        # Original grandstands, pit garages and directional boards.
        for index, fraction in enumerate((0.015, .18, .42, .67, .88)):
            x, y, angle = self.track.pose(self.track.start_progress+fraction*self.track.total_length)
            offset = float(self.track.widths[int(fraction*n)%n]/2)+self.track.RUNOFF_WIDTH+8
            x, y = x-math.sin(angle)*offset, y+math.cos(angle)*offset
            px, py = point((x, y))
            if index == 0:
                pygame.draw.rect(surface, (26, 47, 49), (px-48, py-26, 96, 52))
                pygame.draw.rect(surface, (211, 218, 196), (px-48, py-26, 96, 9))
                for garage in range(4):
                    pygame.draw.rect(surface, (102, 125, 122), (px-43+garage*23, py-7, 18, 24))
                    pygame.draw.rect(surface, LIME, (px-43+garage*23, py-15, 18, 3))
            else:
                pygame.draw.rect(surface, (28, 52, 52), (px-32, py-20, 64, 40))
                for row in range(4):
                    for seat in range(9):
                        pygame.draw.rect(surface, (188, 166, 106) if (seat+row)%3 else (91, 127, 142), (px-29+seat*7, py-17+row*9, 5, 5))
                pygame.draw.rect(surface, (200, 215, 195), (px-35, py-24, 70, 5))
            bx, by = point((x+12, y-7))
            pygame.draw.rect(surface, INK, (bx, by, 26, 15))
            tx, ty = math.cos(angle), math.sin(angle)
            cx, cy = bx+13, by+7
            end = (round(cx+tx*8), round(cy+ty*8))
            pygame.draw.line(surface, LIME, (round(cx-tx*7), round(cy-ty*7)), end, 2)
            pygame.draw.lines(surface, LIME, False,
                              [(round(end[0]-tx*4-ty*3), round(end[1]-ty*4+tx*3)), end,
                               (round(end[0]-tx*4+ty*3), round(end[1]-ty*4-tx*3))], 2)
        self.chunks[key] = surface
        while len(self.chunks) > 128:
            self.chunks.popitem(last=False)
        return surface

    def draw(self, surface, camera, quality):
        x0, y0 = camera.world(0, 0)
        x1, y1 = camera.world(1280, 720)
        unit = self.CHUNK_METERS
        for gx in range(math.floor(x0/unit), math.floor(x1/unit)+1):
            for gy in range(math.floor(y0/unit), math.floor(y1/unit)+1):
                image = self._chunk(gx, gy, quality)
                left, top = camera.screen(gx*unit, gy*unit)
                right, bottom = camera.screen((gx+1)*unit, (gy+1)*unit)
                surface.blit(pygame.transform.scale(image, (right-left, bottom-top)), (left, top))


class Renderer:
    def __init__(self):
        self.type = Typography()
        self.arts = {}
        self.rotations = OrderedDict()
        self.tire_marks = deque(maxlen=700)
        self.particles = deque(maxlen=100)
        self.random = random.Random(18)
        self.car_sprite = self._car()
        self.last_mark = None

    def _car(self):
        sprite = pygame.Surface((80, 38), pygame.SRCALPHA)
        pygame.draw.rect(sprite, (7, 12, 15), (18, 1, 12, 10))
        pygame.draw.rect(sprite, (7, 12, 15), (18, 27, 12, 10))
        pygame.draw.rect(sprite, (7, 12, 15), (59, 1, 10, 10))
        pygame.draw.rect(sprite, (7, 12, 15), (59, 27, 10, 10))
        for y in (7, 30):
            pygame.draw.line(sprite, (127, 145, 150), (25, y), (60, 19), 2)
        pygame.draw.rect(sprite, (213, 234, 209), (10, 5, 6, 28))
        pygame.draw.rect(sprite, (21, 44, 43), (12, 7, 3, 24))
        pygame.draw.polygon(sprite, (196, 240, 79), [(18, 12), (34, 8), (50, 11), (72, 16), (72, 22), (50, 27), (34, 30), (18, 26)])
        pygame.draw.polygon(sprite, (115, 167, 62), [(18, 22), (40, 25), (62, 22), (50, 27), (34, 30), (18, 26)])
        pygame.draw.rect(sprite, (223, 237, 213), (71, 6, 5, 26))
        pygame.draw.rect(sprite, (49, 72, 65), (73, 8, 2, 22))
        pygame.draw.ellipse(sprite, (15, 34, 37), (36, 12, 18, 14))
        pygame.draw.ellipse(sprite, (245, 112, 88), (42, 15, 7, 7))
        pygame.draw.line(sprite, WHITE, (52, 19), (69, 19), 2)
        pygame.draw.rect(sprite, WHITE, (25, 17, 6, 3))
        return sprite

    def clear_effects(self):
        self.tire_marks.clear()
        self.particles.clear()
        self.last_mark = None

    def effects(self, vehicle, state, dt, quality):
        if quality == "low":
            return
        if abs(vehicle.lateral_velocity) > 3 and vehicle.speed > 8:
            if self.last_mark and math.hypot(vehicle.x-self.last_mark[0], vehicle.y-self.last_mark[1]) < 5:
                self.tire_marks.append((self.last_mark, (vehicle.x, vehicle.y)))
            self.last_mark = (vehicle.x, vehicle.y)
        else:
            self.last_mark = None
        if state.offtrack and vehicle.speed > 5:
            self.particles.append([vehicle.x, vehicle.y, self.random.uniform(-2, 2), self.random.uniform(-2, 2), .6])
        for particle in self.particles:
            particle[0] += particle[2]*dt
            particle[1] += particle[3]*dt
            particle[4] -= dt
        while self.particles and self.particles[0][4] <= 0:
            self.particles.popleft()

    def race(self, surface, track, vehicle, state, camera, settings, best, expanded, debug, metrics):
        if track.id not in self.arts:
            self.arts[track.id] = CircuitArt(track)
        self.arts[track.id].draw(surface, camera, settings["quality"])
        for a, b in self.tire_marks:
            pygame.draw.line(surface, (33, 39, 40), camera.screen(*a), camera.screen(*b), 2)
        for x, y, _, _, lifetime in self.particles:
            pygame.draw.circle(surface, (178, 166, 108), camera.screen(x, y), max(1, int(lifetime*7)))
        key = (round(math.degrees(vehicle.heading)/2)*2, round(camera.ppm*6)/6)
        if key not in self.rotations:
            width, height = round(6.6*key[1]), round(3.1*key[1])
            scaled = pygame.transform.scale(self.car_sprite, (width, height))
            self.rotations[key] = pygame.transform.rotate(scaled, -key[0])
            if len(self.rotations) > 360:
                self.rotations.popitem(last=False)
        image = self.rotations[key]
        position = camera.screen(vehicle.x, vehicle.y)
        shadow = pygame.mask.from_surface(image).to_surface(setcolor=(21, 34, 29, 100), unsetcolor=(0, 0, 0, 0))
        surface.blit(shadow, shadow.get_rect(center=(position[0]+3, position[1]+4)))
        surface.blit(image, image.get_rect(center=position))
        if debug:
            self.debug_geometry(surface, track, vehicle, state, camera)
        self.hud(surface, track, state, best)
        self.minimap(surface, track, (1048, 125, 204, 150), vehicle, compact=True)
        if state.countdown > 0:
            overlay = pygame.Surface((250, 190), pygame.SRCALPHA)
            overlay.fill((*INK, 225))
            surface.blit(overlay, (515, 245))
            self.type.text(surface, "LIGHTS OUT IN", (640, 266), 18, LIME, True, "center")
            self.type.text(surface, str(math.ceil(state.countdown)), (640, 286), 94, WHITE, True, "center", True)
            self.type.text(surface, "W / UP  ACCELERATE", (640, 400), 15, MUTED, align="center")
        if expanded:
            surface.fill(INK)
            self.type.text(surface, track.name.upper(), (64, 42), 32, WHITE, True)
            self.type.text(surface, "CIRCUIT OVERVIEW   /   TAB TO CLOSE", (64, 84), 16, LIME)
            self.minimap(surface, track, (160, 130, 960, 500), vehicle, compact=False, checkpoints=True)
        if debug:
            v = state.vehicle
            rows = [f"FPS {metrics[0]:5.1f}   STEPS/S {metrics[1]:5.0f}",
                    f"POS {v.x:8.2f}, {v.y:8.2f} m", f"HEADING {math.degrees(v.heading):7.2f} deg",
                    f"FORWARD {v.longitudinal_velocity:7.2f} m/s", f"LATERAL {v.lateral_velocity:7.2f} m/s",
                    f"STEER {math.degrees(v.steering_angle):7.2f} deg", f"SURFACE {state.surface}",
                    f"OFFSET {state.lateral_distance:7.2f} m", f"PROGRESS {state.track_progress:7.2f} m",
                    f"CHECKPOINT {state.next_checkpoint}/{len(track.checkpoints)}", f"COLLISION {state.collision}"]
            panel = pygame.Surface((326, 268), pygame.SRCALPHA)
            panel.fill((*INK, 235))
            surface.blit(panel, (24, 142))
            for i, row in enumerate(rows):
                self.type.text(surface, row, (38, 153+i*23), 15, LIME if i == 0 else WHITE, mono=True)

    def hud(self, surface, track, state, best):
        pygame.draw.rect(surface, INK, (0, 0, 1280, 102))
        pygame.draw.rect(surface, LIME, (24, 23, 4, 54))
        self.type.text(surface, "APEX / RL", (42, 22), 25, WHITE, True)
        self.type.text(surface, track.name.upper(), (43, 55), 15, MUTED)
        self.type.text(surface, f"LAP {state.lap_number:02}", (300, 22), 17, LIME, True)
        self.type.text(surface, format_time(state.lap_time), (298, 45), 30, WHITE, mono=True)
        self.type.text(surface, "PERSONAL BEST", (535, 22), 14, MUTED)
        self.type.text(surface, format_time(best), (534, 47), 24, WHITE, mono=True)
        self.type.text(surface, "PREVIOUS LAP", (755, 22), 14, MUTED)
        self.type.text(surface, format_time(state.previous_lap), (754, 47), 24, WHITE, mono=True)
        self.type.text(surface, f"{state.completed_laps} VALID LAPS", (1252, 35), 17, LIME, True, "right")
        pygame.draw.rect(surface, INK, (24, 577, 244, 119))
        self.type.text(surface, f"{round(state.vehicle.speed*3.6):03}", (43, 583), 62, WHITE, True, mono=True)
        self.type.text(surface, "KM/H", (196, 619), 17, MUTED)
        throttle = state.vehicle.throttle_input
        brake = state.vehicle.brake_input
        pygame.draw.rect(surface, (43, 60, 58), (43, 670, 200, 6))
        pygame.draw.rect(surface, RED if brake else LIME, (43, 670, round(200*(brake or throttle)), 6))
        self.type.text(surface, "WASD / ARROWS   DRIVE     R   RECOVER     ESC   PAUSE", (1252, 675), 14, WHITE, align="right")
        if state.offtrack or not state.lap_valid:
            message = "OFF TRACK / REDUCED GRIP" if state.offtrack else "LAP INVALID / "+state.invalid_reason.upper()
            pygame.draw.rect(surface, INK, (340, 624, 600, 39))
            self.type.text(surface, message, (640, 635), 16, RED, True, "center")

    def minimap(self, surface, track, rect, vehicle=None, compact=False, checkpoints=False):
        rect = pygame.Rect(rect)
        pygame.draw.rect(surface, PANEL, rect, border_radius=8)
        inner = rect.inflate(-36, -36)
        x0, y0, x1, y1 = track.bounds
        scale = min(inner.width/(x1-x0), inner.height/(y1-y0))
        def transform(x, y):
            return round(rect.centerx+(x-(x0+x1)/2)*scale), round(rect.centery+(y-(y0+y1)/2)*scale)
        points = [transform(*p) for p in track.points[::3]]
        pygame.draw.lines(surface, (60, 80, 77), True, points, 7 if not compact else 5)
        pygame.draw.lines(surface, WHITE, True, points, 3 if not compact else 2)
        finish = track.finish
        nx, ny = -finish.tangent[1], finish.tangent[0]
        extent = 8/scale
        pygame.draw.line(surface, RED, transform(finish.center[0]+nx*extent, finish.center[1]+ny*extent), transform(finish.center[0]-nx*extent, finish.center[1]-ny*extent), 3)
        if checkpoints:
            for index, gate in enumerate(track.checkpoints):
                p = transform(*gate.center)
                pygame.draw.circle(surface, (91, 136, 125), p, 5)
                self.type.text(surface, index+1, (p[0]+9, p[1]-9), 13, MUTED, mono=True)
        if vehicle:
            p = transform(vehicle.x, vehicle.y)
            pygame.draw.circle(surface, INK, p, 7)
            pygame.draw.circle(surface, LIME, p, 4)
            pygame.draw.line(surface, LIME, p, (p[0]+int(math.cos(vehicle.heading)*11), p[1]+int(math.sin(vehicle.heading)*11)), 2)

    def debug_geometry(self, surface, track, vehicle, state, camera):
        for path, color in ((track.points, (240, 216, 107)), (track.left_boundary, (76, 182, 213)), (track.right_boundary, (76, 182, 213))):
            pygame.draw.lines(surface, color, True, [camera.screen(*p) for p in path[::4]], 1)
        from apexrl.core.vehicle import VehicleConfig, VehicleState
        car = VehicleState(x=vehicle.x, y=vehicle.y, heading=vehicle.heading)
        pygame.draw.lines(surface, RED, True, [camera.screen(*p) for p in car.footprint(self.vehicle_config)], 1)
        p = track.project(vehicle.x, vehicle.y)
        pygame.draw.line(surface, LIME, camera.screen(vehicle.x, vehicle.y), camera.screen(p.x, p.y), 2)
        pygame.draw.line(surface, LIME, camera.screen(p.x, p.y), camera.screen(p.x+p.tangent[0]*12, p.y+p.tangent[1]*12), 2)
        for gate in track.checkpoints:
            nx, ny = -gate.tangent[1], gate.tangent[0]
            pygame.draw.line(surface, RED, camera.screen(gate.center[0]+nx*gate.half_width, gate.center[1]+ny*gate.half_width), camera.screen(gate.center[0]-nx*gate.half_width, gate.center[1]-ny*gate.half_width), 1)
