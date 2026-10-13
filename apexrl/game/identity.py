"""Original circuit-inspired app icon; no external image or font assets."""
import pygame


def app_icon(size: int = 64) -> pygame.Surface:
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    surface.fill((13,24,27))
    def points(values):
        return [(round(x*size), round(y*size)) for x,y in values]
    pygame.draw.polygon(surface, (205,247,92), points(((.12,.82),(.43,.16),(.60,.16),(.89,.82),(.69,.82),(.51,.37),(.32,.82))))
    pygame.draw.polygon(surface, (235,241,225), points(((.39,.63),(.64,.63),(.69,.75),(.34,.75))))
    return surface
