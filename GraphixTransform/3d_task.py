from graphics import *
import numpy as np
import math as mt
import time
import itertools

# ---------------- Параметри вікна та об'єкта ----------------
WIN_W, WIN_H = 700, 700
CENTER = np.array([WIN_W/2.0, WIN_H/2.0])
PYR_SIZE = 200.0
FPS_DELAY = 0.02

# ------------------ Вхідна (розширена) матриця вершин піраміди ------------------
h = 220.0
s = PYR_SIZE

# Трикутник основи
# базова 2D трикутна рамка
a = s / mt.sqrt(3)
V_base = [
    (-s/2.0, -a/3.0, -h/2.0, 1),
    ( s/2.0, -a/3.0, -h/2.0, 1),
    ( 0.0,    2*a/3.0, -h/2.0, 1),
]

V_top = (0.0, 0.0, h/2.0, 1)

Vertices = np.array(V_base + [V_top], dtype=float)

# ------------------ Грани (трикутники) для малювання (індекси вершин) -----------
FACES = [
    (0,1,2),
    (0,1,3),
    (1,2,3),
    (2,0,3)
]

COLOR_PALETTE = ["#ff9999", "#ffd699", "#ffff99", "#c9ff99", "#99ffd6", "#99e6ff", "#cba3ff"]

# ------------------ Утиліти: матриці та застосування ------------------
def translation(tx, ty, tz):
    M = np.eye(4)
    M[0,3] = tx
    M[1,3] = ty
    M[2,3] = tz
    return M

def rotate_about_axis_through_point(point, axis, angle_deg):
    """
    Повертає 4x4 матрицю обертання навколо прямої, що проходить через 'point'
    у напрямку одиничного вектора 'axis' (ux,uy,uz), на кут angle_deg (градуси).
    Використовує Rodrigues для 3x3, вбудовує в 4x4, з перекладом.
    """
    ux, uy, uz = axis / np.linalg.norm(axis)
    theta = mt.radians(angle_deg)
    c = mt.cos(theta)
    s = mt.sin(theta)
    R = np.array([
        [c + ux*ux*(1-c),      ux*uy*(1-c) - uz*s,  ux*uz*(1-c) + uy*s],
        [uy*ux*(1-c) + uz*s,   c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
        [uz*ux*(1-c) - uy*s,   uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)]
    ], dtype=float)
    M = np.eye(4)
    M[:3,:3] = R
    T1 = translation(-point[0], -point[1], -point[2])
    T2 = translation(point[0], point[1], point[2])
    M_col = T2 @ M @ T1
    return M_col

def rotation_x(deg):
    a = mt.radians(deg)
    R = np.eye(4)
    R[1,1] = mt.cos(a); R[1,2] = -mt.sin(a)
    R[2,1] = mt.sin(a); R[2,2] =  mt.cos(a)
    return R

def rotation_y(deg):
    a = mt.radians(deg)
    R = np.eye(4)
    R[0,0] =  mt.cos(a); R[0,2] = mt.sin(a)
    R[2,0] = -mt.sin(a); R[2,2] = mt.cos(a)
    return R

def rotation_z(deg):
    a = mt.radians(deg)
    R = np.eye(4)
    R[0,0] = mt.cos(a); R[0,1] = -mt.sin(a)
    R[1,0] = mt.sin(a); R[1,1] =  mt.cos(a)
    return R

def orthographic_projection_matrix():
    """Проста ортографічна проекція: z відкидається"""
    P = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,0,0],
        [0,0,0,1]
    ], dtype=float)
    return P

def apply_transform(V, M):
    """
    Застосувати 4x4 матрицю M до матриці вершин V (N x 4).
    Ми використовуємо row-vector convention: V_out = V @ M.T
    """
    return V.dot(M.T)

def to_screen_points(V_transformed, center=CENTER):
    P = orthographic_projection_matrix()
    Vp = apply_transform(V_transformed, P)
    pts2d = []
    for v in Vp:
        x = center[0] + v[0]
        y = center[1] - v[1]
        pts2d.append( (x, y) )
    return pts2d, Vp

# ------------------ Візуалізація: малювання граней згідно глибини ------------------
def draw_pyramid(win, V_world, faces=FACES, face_colors=None):
    pts2d, Vp = to_screen_points(V_world)
    face_depth = []
    for i, face in enumerate(faces):
        zs = [V_world[idx, 2] for idx in face]
        face_depth.append((np.mean(zs), i))
    face_depth.sort()
    drawn = []
    for _, fi in face_depth:
        face = faces[fi]
        poly_pts = [Point(*pts2d[idx]) for idx in face]
        poly = Polygon(*poly_pts)
        if face_colors:
            poly.setFill(face_colors[fi % len(face_colors)])
        poly.setOutline("black")
        poly.draw(win)
        drawn.append(poly)
    return drawn

# ------------------ Функція для створення interpolated color (hex) ------------------
def interp_color_hex(hex1, hex2, t):
    """Інтерполює між двома hex-кольорами за параметром t∈[0,1]"""
    def hex_to_rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2],16) for i in (0,2,4))
    def rgb_to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(*[int(max(0,min(255,round(c)))) for c in rgb])
    c1 = np.array(hex_to_rgb(hex1), dtype=float)
    c2 = np.array(hex_to_rgb(hex2), dtype=float)
    c = (1-t)*c1 + t*c2
    return rgb_to_hex(tuple(c))

# ------------------ Головна анімація ------------------
def main_animation():
    win = GraphWin("3D Pyramid - axonometric + rotation", WIN_W, WIN_H, autoflush=False)
    win.setBackground("white")
    centroid = np.mean(Vertices[:,:3], axis=0)
    axis_dir = np.array([1.0, 0.7, 0.4])
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    iso_rx = 35.2643897
    iso_rz = 45.0
    Iso = rotation_z(iso_rz) @ rotation_x(iso_rx)

    total_cycles = 3
    frames_per_cycle = 240
    appear_frames = 40
    hold_frames = 80
    fade_frames = 40
    rotate_frames = frames_per_cycle
    color_cycle = itertools.cycle(COLOR_PALETTE)

    T_to_origin = translation(-centroid[0], -centroid[1], -centroid[2])
    drawn_objects = []

    try:
        for cycle in range(total_cycles):
            face_base = [COLOR_PALETTE[(cycle + i) % len(COLOR_PALETTE)] for i in range(len(FACES))]
            for frame in range(frames_per_cycle):
                for obj in drawn_objects:
                    obj.undraw()

                if frame < appear_frames:
                    t_alpha = frame / max(1, appear_frames-1)
                elif frame < appear_frames + hold_frames:
                    t_alpha = 1.0
                elif frame < appear_frames + hold_frames + fade_frames:
                    t_alpha = 1.0 - (frame - (appear_frames + hold_frames)) / max(1, fade_frames-1)
                else:
                    t_alpha = 0.0

                shift = (frame * 3) % len(COLOR_PALETTE)
                face_colors = []
                for i, base in enumerate(face_base):
                    other = COLOR_PALETTE[(i + shift) % len(COLOR_PALETTE)]
                    pulse = 0.5*(1 + mt.sin(2*mt.pi*(frame/rotate_frames) + i))
                    mixed = interp_color_hex(base, other, pulse)
                    final_color = interp_color_hex("#ffffff", mixed, t_alpha)
                    face_colors.append(final_color)

                angle_deg = (360.0 * frame) / rotate_frames
                R_axis = rotate_about_axis_through_point(centroid, axis_dir, angle_deg)

                V_trans = apply_transform(Vertices, T_to_origin)
                V_trans = apply_transform(V_trans, R_axis)
                V_trans = apply_transform(V_trans, Iso)

                drawn_objects = draw_pyramid(win, V_trans, faces=FACES, face_colors=face_colors)
                update()
                time.sleep(FPS_DELAY)

            time.sleep(0.15)

        win.getMouse()
    finally:
        win.close()

if __name__ == "__main__":
    main_animation()
