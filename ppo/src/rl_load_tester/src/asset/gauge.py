import math
import pygame
import pygame.gfxdraw


class Gauge:

    def __init__(self, screen, FONT, x_cord, y_cord, thickness, radius, circle_colour, glow=False):
        self.screen = screen
        self.Font = FONT
        self.x_cord = x_cord
        self.y_cord = y_cord
        self.thickness = thickness
        self.radius = radius
        self.circle_colour = circle_colour
        self.glow = glow

    def draw(self, percent):
        fill_angle = int(percent * 270 / 100)
        per = percent
        if percent > 100:
            percent = 100
        if per < 0:
            per = 0
        if per > 100:
            per = 100

        ac = [int(255 - per * 255 / 100), int(per * 255 / 100), int(0), 255]

        for indexi in range(len(ac)):
            if ac[indexi] < 0:
                ac[indexi] = 0
            if ac[indexi] > 255:
                ac[indexi] = 255

        if percent <= 50:
            # Green: [0, 255, 0]
            ac = [0, 255, 0, 255]
        elif percent <= 79:
            # Orange: [255, 165, 0]
            ac = [255, 165, 0, 255]
        else:
            # Red: [255, 0, 0]
            ac = [255, 0, 0, 255]

        pertext = self.Font.render(str(percent) + "%", True, ac)
        pertext_rect = pertext.get_rect(
            center=(int(self.x_cord), int(self.y_cord)))
        self.screen.blit(pertext, pertext_rect)
