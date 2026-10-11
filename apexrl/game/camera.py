import math


class Camera:
    def __init__(self):
        self.x = self.y = 0.0
        self.ppm = 5.0
        self.ready = False

    def reset(self, vehicle):
        self.x, self.y = vehicle.x, vehicle.y
        self.ready = False

    def update(self, vehicle, dt: float, smoothing: float, base_ppm: float):
        target_scale = base_ppm*(1-0.18*min(1, vehicle.speed/83.3))
        ahead = 12+vehicle.speed*0.35
        tx = vehicle.x+math.cos(vehicle.heading)*ahead
        ty = vehicle.y+math.sin(vehicle.heading)*ahead
        if not self.ready:
            self.x, self.y, self.ppm, self.ready = tx, ty, target_scale, True
        weight = 1-math.exp(-smoothing*min(dt, 0.1))
        self.x += (tx-self.x)*weight
        self.y += (ty-self.y)*weight
        self.ppm += (target_scale-self.ppm)*weight

    def screen(self, x, y):
        return round((x-self.x)*self.ppm+640), round((y-self.y)*self.ppm+390)

    def world(self, x, y):
        return (x-640)/self.ppm+self.x, (y-390)/self.ppm+self.y
