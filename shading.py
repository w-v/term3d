import time
from typing import List
import numpy as np
from scipy.spatial.transform import Rotation


def gen_cube():
    n = 40
    d = np.linspace(0, 1, n)
    g = np.array(np.meshgrid(d, d)).swapaxes(0, 2)
    g0 = np.concatenate([g, np.zeros((*g.shape[:-1], 1))], axis=-1).reshape(
        -1, 3
    )
    g1 = np.concatenate([g, np.ones((*g.shape[:-1], 1))], axis=-1).reshape(
        -1, 3
    )
    cube = np.stack(
        [
            g0,
            g0[..., [2, 0, 1]],
            g0[..., [0, 2, 1]],
            g1,
            g1[..., [2, 0, 1]],
            g1[..., [0, 2, 1]],
        ]
    )

    normals = np.array(
        [
            [0, 0, -1],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ]
    )
    normals = (
        np.full((cube.shape[1], cube.shape[0], 3), normals)
        .swapaxes(0, 1)
        .reshape((-1, 3))
    )
    cube = cube.reshape((-1, 3))
    cube -= np.array([0.5, 0.5, 0.5])
    return [cube, normals]


def render(obj, a):
    def getch(x, y):
        pass

    K = 30
    W = 80
    H = 40
    points, normals = obj
    # rot = np.identity(3)
    rot = Rotation.from_euler(
        "xyz", [45 + 2 * a, 45 + a, 0], degrees=True
    ).as_matrix()
    pos = np.array([0, 0, 1.5])
    points = np.einsum("...ij,...j", rot, points) + pos
    oz = 1 / points[:, 2]
    nverts = points[:, [0, 1]]
    nverts[:, 0] = W / 2 + K * oz * nverts[:, 0]
    nverts[:, 1] = H / 2 - K * oz * nverts[:, 1] * 0.55

    coords = np.round(nverts).astype(int)

    light_dir = np.array([0, -1, 1]) * -1
    light_dir = light_dir / np.linalg.norm(light_dir)

    normals = normals / np.linalg.norm(normals, axis=-1)[..., np.newaxis]
    normals = np.einsum("...ij,...j", rot, normals)
    lum = np.dot(normals, light_dir)
    facing = lum >= -0.5
    coords = coords[facing]
    oz = oz[facing]
    lum_id = np.clip(np.round(((lum + 0.5) / 1.5) * 11).astype(int), 0, 11)

    target = np.full((W, H), -1, dtype=int)
    order = np.argsort(oz)
    lum_id = lum_id[order]
    coords = coords[order]
    # zbuf = np.zeros((W, H))

    # import pdb

    # pdb.set_trace()

    # for c, z, l in zip(coords, oz, lum_id):
    #     if z > zbuf[(*c,)]:
    #         target[(*c,)] = l
    #         zbuf[(*c,)] = z

    coords = coords.swapaxes(0, 1)
    coords = (np.array(coords[0]), np.array(coords[1]))

    target[coords] = lum_id

    # import pdb

    # pdb.set_trace()

    chars = "|\n".join(
        ["_" * W]
        + [
            "".join(
                [
                    ".,-~:;=!*#$@"[target[i, j]] if target[i, j] >= 0 else " "
                    for i in range(W)
                ]
            )
            for j in range(H)
        ]
        + ["_" * W]
    )
    print("\x1b[H")
    print(chars)


def main():
    # a = np.linspace(-1, 1, 10)
    # b = np.vstack([a, a, np.zeros(10)]).swapaxes(0, 1)
    # obj = [b, b]

    # d = np.linspace(-1, 1, 10)
    # g = np.array(np.meshgrid(d, d)).swapaxes(0, 2)
    # g = np.concatenate([g, np.zeros((10, 10, 1))], axis=-1).reshape(-1, 3)
    # n = np.full(g.shape, np.array([0, 0, -1]))
    # obj = [g, n]

    # import pdb

    # pdb.set_trace()
    obj = gen_cube()
    a = 0
    while True:
        t = time.time()
        a += 0.3
        render(obj, a)
        print("%.2f fps" % (1 / (time.time() - t)))


if __name__ == "__main__":
    main()
