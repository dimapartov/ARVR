import numpy as np
from PIL import Image
from collections import deque

def count_shapes_bw(image_path, circle_radius, square_side):
    img = Image.open(image_path).convert('L')
    mask = np.array(img) < 128
    return _count_shapes(mask, circle_radius, square_side)

def count_shapes_color(image_path, circle_radius, square_side, color):
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img)
    mask = (arr[:,:,0] == color[0]) & (arr[:,:,1] == color[1]) & (arr[:,:,2] == color[2])
    return _count_shapes(mask, circle_radius, square_side)

def _count_shapes(mask: np.ndarray, circle_radius: int, square_side: int) -> tuple:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    expected_bbox_circle = 2 * circle_radius + 1
    expected_bbox_square = square_side + 1
    circles = squares = 0

    for i in range(h):
        for j in range(w):
            if mask[i, j] and not visited[i, j]:
                minr = maxr = i
                minc = maxc = j
                queue = deque([(i, j)])
                visited[i, j] = True

                while queue:
                    x, y = queue.popleft()
                    minr, maxr = min(minr, x), max(maxr, x)
                    minc, maxc = min(minc, y), max(maxc, y)
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            nx, ny = x+dx, y+dy
                            if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not visited[nx, ny]:
                                visited[nx, ny] = True
                                queue.append((nx, ny))

                height = maxr - minr + 1
                width = maxc - minc + 1

                if minr == 0 or minc == 0 or maxr == h-1 or maxc == w-1:
                    continue

                sub = mask[minr:maxr+1, minc:maxc+1]
                if height == expected_bbox_square and width == expected_bbox_square and sub.all():
                    squares += 1
                elif height == expected_bbox_circle and width == expected_bbox_circle:
                    circles += 1

    return circles, squares

if __name__ == "__main__":
    bw_c, bw_s = count_shapes_bw('images/8_1.png', circle_radius=10, square_side=20)
    print(f"BW image -> Circles: {bw_c}, Squares: {bw_s}")

    col_c, col_s = count_shapes_color('images/8_2.png', circle_radius=10, square_side=20, color=(255, 0, 0))
    print(f"Color image -> Circles: {col_c}, Squares: {col_s}")