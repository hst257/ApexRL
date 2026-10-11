"""Original short synthesized sounds; no external assets or downloads."""
import math
import numpy as np
import pygame


class Audio:
    def __init__(self):
        self.available = False
        self.engine_channel = None
        self.engine_index = -1
        self.engine_bank = []
        self.sounds = {}
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            self.engine_channel = pygame.mixer.Channel(0)
            self.effect_channel = pygame.mixer.Channel(1)
            for frequency in range(60, 361, 20):
                # Integer-cycle loops avoid clicks at their boundaries.
                t = np.arange(22050)/22050
                wave = (np.sin(2*math.pi*frequency*t)+0.28*np.sin(4*math.pi*frequency*t))*0.16
                self.engine_bank.append(self._sound(wave))
            for name, frequency, duration in (("click", 650, .055), ("count", 540, .16), ("go", 920, .24), ("lap", 1150, .3), ("collision", 90, .15)):
                t = np.arange(int(22050*duration))/22050
                wave = np.sin(2*math.pi*frequency*t)*np.exp(-t*12)*0.3
                self.sounds[name] = self._sound(wave)
            self.available = True
        except pygame.error:
            pass

    def _sound(self, wave):
        samples = np.asarray(wave*32767, dtype=np.int16)
        if pygame.mixer.get_init()[2] == 2:
            samples = np.column_stack((samples, samples))
        return pygame.sndarray.make_sound(np.ascontiguousarray(samples))

    def effect(self, name, volume):
        if self.available:
            self.effect_channel.set_volume(volume*0.6)
            self.effect_channel.play(self.sounds[name])

    def update(self, vehicle, volume, active):
        if not self.available:
            return
        if not active:
            self.engine_channel.stop()
            self.engine_index = -1
            return
        index = min(len(self.engine_bank)-1, int(vehicle.speed/6+vehicle.throttle_input*2))
        if index != self.engine_index:
            self.engine_channel.play(self.engine_bank[index], loops=-1, fade_ms=35)
            self.engine_index = index
        self.engine_channel.set_volume(volume*(0.2+vehicle.throttle_input*0.35))

    def close(self):
        if self.available:
            pygame.mixer.stop()
