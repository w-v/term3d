import time
from typing import List
import numpy as np
from scipy.spatial.transform import Rotation

import pynimation.common.data as data_
from pynimation.anim.animation import Animation


def gen_anim(file="data/animations/walk_loop.bvh"):
    bvh_file = data_.getDataPath(file)
    animation = Animation.load(bvh_file)
    animation.root.pos.z = 0

    # animation.setFramerate(30)
    lines = []
    for a in animation.skeleton.joints:
        if a.parent is not None:
            lines.append([a.parent.id, a.id])

    return ([animation.globals.pos, np.array(lines)], animation)


def projection(
    view_width: int,
    view_height: int,
    fov: int,
    near: float,
    far: int,
):
    aspect = view_width / view_height
    range_ = np.tan(np.radians(fov) / 2.0)

    sx = 1 / (aspect * range_)
    sy = 1 / range_
    sz = -(far + near) / (far - near)
    pz = -(2.0 * far * near) / (far - near)

    proj = np.zeros((4, 4))

    proj[0, 0] = sx
    proj[1, 1] = sy
    proj[2, 2] = sz
    proj[2, 3] = pz
    proj[3, 2] = -1.0

    return proj


def gen_cube():
    verts = np.zeros((8, 3))
    verts[1, 0] += 1
    verts[2, [0, 2]] += 1
    verts[3, 2] += 1
    verts[4:] = verts[:4]
    verts[4:, 1] += 1

    verts -= np.array([0.5, 0.5, 0.5])

    lines = np.zeros((12, 2), dtype=int)
    a = np.arange(4)
    lines[:4, 0] = a
    lines[:4, 1] = np.roll(a, -1)
    lines[4:8, 0] = a + 4
    lines[4:8, 1] = np.roll(a + 4, -1)
    lines[8:] = np.array(list(zip(a, a + 4)))

    return [verts, lines]


def gen_line():
    verts = np.zeros((2, 3))
    verts[0] = np.array([0, 0, 0])
    verts[1] = np.array([1, 0.07, 0])

    lines = np.zeros((1, 2), dtype=int)
    lines[0] = np.array([0, 1])

    return [verts, lines]


def render(obj: List[np.ndarray], a):
    W = 50
    H = 50
    model, view, proj = [np.identity(4) for _ in range(3)]
    pos = np.array([0, -1, 2])
    model[:3, 3] += pos
    scale = np.identity(3)
    # model[:3, :3] = (
    #     # Rotation.from_euler("XYZ", [45, 45 + a, 0], degrees=True).as_matrix()
    #     Rotation.from_rotvec(
    #         np.array([-1, 0, 1]) * np.radians(-30)
    #     ).as_matrix()
    #     @ scale
    # )
    # model[:3, :3] = [
    #     [0.78867513, -0.57735027, -0.21132487],
    #     [0.57735027, 0.57735027, 0.57735027],
    #     [-0.21132487, -0.57735027, 0.78867513],
    # ]
    model[:3, :3] = Rotation.from_euler(
        "xyz", [0, 180 + a, 0], degrees=True
    ).as_matrix()
    view = np.identity(4)
    proj = projection(view_width=W, view_height=H, fov=30, near=0.001, far=100)
    m = proj @ view @ model
    verts, lines = obj
    hverts = np.hstack([verts, np.ones(len(verts)).reshape((len(verts), 1))])
    # hverts = np.einsum("...ij,...j", model, hverts)
    nverts = np.einsum("...ij,...j", m, hverts)
    nverts = nverts / nverts[:, 3][:, np.newaxis]
    nverts = nverts[:, [0, 1]]
    nverts[:, 0] *= -1
    nverts = (nverts + 1) * 0.5
    nverts[:, 0] *= W
    nverts[:, 1] *= H * 0.55
    nverts = np.round(nverts)
    # nverts = np.clip(nverts, -1, 1)

    # np.set_printoptions(suppress=True, precision=5)
    # print(nverts)

    target = np.zeros((W, H), dtype=bool)

    for line in lines:
        f, t = nverts[line]
        if np.all(f == t):
            x = np.array([f[0]])
            y = np.array([f[1]])
        else:
            if abs(t[0] - f[0]) >= abs(t[1] - f[1]):
                sign = int(np.sign(t[0] - f[0]))
                if sign == 0:
                    sign = 1
                x = np.arange(f[0], t[0] + sign, sign, dtype=int)
                y = f[1] + ((x - f[0]) / (t[0] - f[0])) * (t[1] - f[1])
            else:
                sign = int(np.sign(t[1] - f[1]))
                if sign == 0:
                    sign = 1
                y = np.arange(f[1], t[1] + sign, sign, dtype=int)
                x = f[0] + ((y - f[1]) / (t[1] - f[1])) * (t[0] - f[0])

        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)
        coords = np.vstack([x, y]).astype(int).tolist()
        coords = (np.array(coords[0]), np.array(coords[1]))
        target[coords] = True

    chars = "\n".join(
        [
            "".join(["#" if target[i, j] else " " for i in range(W)])
            for j in range(H)
        ]
    )
    print(chars)


def main():
    # cube = gen_cube()
    # obj = gen_cube()
    # obj = gen_cube()
    # a = 0
    # # render(obj, a)
    # while True:
    #     a += 0.2
    #     render(obj, a)
    i = 0
    a = 0
    ([verts, lines], animation) = gen_anim()
    while True:
        a += 0.2
        fid = animation.idAt(time.time())
        render([verts[fid], lines], a)
        i += 1


if __name__ == "__main__":
    main()
