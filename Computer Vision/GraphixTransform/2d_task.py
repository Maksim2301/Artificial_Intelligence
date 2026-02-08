from graphics import *
import numpy as np
import math as mt
import time

# ---------------- Параметри ----------------
WIN_W, WIN_H = 600, 600
SQUARE_SIZE = 80
DX, DY = 5, -5
ANGLE_STEP = 30
STEPS = 90
DELAY = 0.1

# ---------------- Допоміжні функції ----------------
def translate(point, dx, dy):
    """Перенесення точки матрично"""
    a = np.array([[point[0], point[1], 1]])
    f = np.array([[1, 0, dx],
                  [0, 1, dy],
                  [0, 0, 1]])
    return (a @ f.T)[0, :2]

def rotate(point, angle_deg, center):
    """Обертання точки матрично навколо довільного центру"""
    cx, cy = center
    angle = mt.radians(angle_deg)

    a = np.array([[point[0] - cx, point[1] - cy, 1]])

    r = np.array([[mt.cos(angle), -mt.sin(angle), 0],
                  [mt.sin(angle),  mt.cos(angle), 0],
                  [0, 0, 1]])
    total = a @ r.T

    return total[0, 0] + cx, total[0, 1] + cy

def draw_square(win, points, color="blue"):
    """Малює квадрат за списком точок"""
    poly = Polygon(*[Point(x, y) for x, y in points])
    poly.setOutline(color)
    poly.setWidth(2)
    poly.draw(win)
    return poly

# ---------------- Основна програма ----------------
def main():
    win = GraphWin("2D Перетворення: обертання та переміщення", WIN_W, WIN_H)
    win.setBackground("white")

    x0, y0 = 20, 500
    square = [(x0, y0),
              (x0 + SQUARE_SIZE, y0),
              (x0 + SQUARE_SIZE, y0 + SQUARE_SIZE),
              (x0, y0 + SQUARE_SIZE)]

    cx = x0 + SQUARE_SIZE / 2
    cy = y0 + SQUARE_SIZE / 2

    # --- Циклічне обертання навколо центру ---
    for step in range(36):
        rotated = [rotate(p, step * 10, center=(cx, cy)) for p in square]
        draw_square(win, rotated, "red")
        time.sleep(DELAY)

    # --- Циклічне обертання + перенесення ---
    angle = 0
    moved_square = square
    moved_cx, moved_cy = cx, cy

    for step in range(STEPS):
        # обертання навколо власного центру
        rotated = [rotate(p, angle, center=(moved_cx, moved_cy)) for p in moved_square]

        draw_square(win, rotated, "green")
        time.sleep(DELAY)

        # переносимо фігуру та її центр
        moved_square = [translate(p, DX, DY) for p in moved_square]
        moved_cx, moved_cy = translate((moved_cx, moved_cy), DX, DY)

        # збільшуємо кут
        angle += ANGLE_STEP

    win.getMouse()
    win.close()


if __name__ == "__main__":
    main()
