import pygame
from apexrl.core.controls import VehicleControl


class KeyboardController:
    def action(self, keys, sensitivity: float) -> VehicleControl:
        left = keys[pygame.K_a] or keys[pygame.K_LEFT]
        right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
        throttle = keys[pygame.K_w] or keys[pygame.K_UP]
        brake = keys[pygame.K_s] or keys[pygame.K_DOWN]
        return VehicleControl((int(right)-int(left))*sensitivity, float(throttle), float(brake))
