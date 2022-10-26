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
    uniq = np.unique(cube, axis=0, return_index=True)[1]
    cube = cube[uniq]
    normals = normals[uniq]
    cube -= np.array([0.5, 0.5, 0.5])
    return [cube, normals]


def render_cube():
    obj = gen_cube()
    # obj = gen_sins()
    a = 0
    # pos = np.array([0, -10, 40])
    pos = np.array([0, 0, 2])
    light_dir = np.array([0, -1, 1]) * -1
    light_dir = light_dir / np.linalg.norm(light_dir)
    while True:
        t = time.time()
        a += 0.2
        rot = Rotation.from_euler(
            "xyz", [45 + 2 * a, 45 + a, 0], degrees=True
        ).as_matrix()
        # obj = gen_sins(a)
        render(obj, rot, pos, light_dir)
        print("%.2f fps" % (1 / (time.time() - t)))


def render(obj, rot, pos, light_dir):
    def getch(x, y):
        pass

    K = 30
    W = 80
    H = 40
    points, normals = obj
    # rot = np.identity(3)
    points = np.einsum("...ij,...j", rot, points) + pos
    oz = 1 / points[:, 2]
    nverts = points[:, [0, 1]]
    nverts[:, 0] = W / 2 + K * oz * nverts[:, 0]
    nverts[:, 1] = H / 2 - K * oz * nverts[:, 1] * 0.55

    coords = np.round(nverts).astype(int)
    clip = (
        (coords[:, 0] < W)
        & (coords[:, 1] < H)
        & (coords[:, 0] > 0)
        & (coords[:, 1] > 0)
        & (oz > 0)
    )
    coords = coords[clip]
    oz = oz[clip]
    normals = normals[clip]

    normals = normals / np.linalg.norm(normals, axis=-1)[..., np.newaxis]
    normals = np.einsum("...ij,...j", rot, normals)
    lum = np.dot(normals, light_dir)
    facing = lum >= -0.5
    coords = coords[facing]
    oz = oz[facing]
    lum_id = np.clip(
        np.round(((lum[facing] + 0.5) / 1.5) * 11).astype(int), 0, 11
    )

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
    # print("\x1b[H")
    print(chars)


def gen_sins(c):
    a, b = 4, 0.2
    d, e = 80, 40
    # n = 80
    dd = np.linspace(-d, d, 3 * d)
    ee = np.linspace(-e, e, 3 * e)
    g = np.array(np.meshgrid(dd, ee)).swapaxes(0, 2)
    gg = np.concatenate(
        [
            g[..., [0]],
            a * np.sin(g[..., [0]] * b) * np.sin(g[..., [1]] * b + c),
            g[..., [1]],
        ],
        axis=-1,
    ).reshape(-1, 3)
    n0 = np.empty(gg.shape)
    n1 = np.empty(gg.shape)

    n0[:, 0] = 1
    n0[:, 1] = a * b * np.cos(b * gg[:, 0]) * np.sin(b * gg[:, 2] + c)
    n0[:, 2] = 0

    n1[:, 0] = 0
    n1[:, 1] = a * b * np.sin(b * gg[:, 0]) * np.cos(b * gg[:, 2] + c)
    n1[:, 2] = 1

    n0 = n0 / np.linalg.norm(n0)
    n1 = n1 / np.linalg.norm(n1)
    n = np.cross(n1, n0)

    return [gg, n]


def render_sins():
    a = 0
    pos = np.array([0, -10, 40])

    light_dir = np.array([-0.6, -1, 1])
    light_dir = light_dir / np.linalg.norm(light_dir)
    light_dir *= -0.9

    while True:
        t = time.time()
        a += 0.05
        rot = Rotation.from_euler("xyz", [-30, 0, 0], degrees=True).as_matrix()
        obj = gen_sins(a)
        render(obj, rot, pos, light_dir)
        print("%.2f fps" % (1 / (time.time() - t)))


def interpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin_noise(
    # shape, res, rng, tileable=(True, True), interpolant=interpolant
    angles,
    grid,
    d,
):
    """Generate a 2D numpy array of perlin noise.
    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).
    Returns:
        A numpy array of shape shape with the generated noise.
    Raises:
        ValueError: If shape is not a multiple of res.
    """
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    # if tileable[0]:
    #     gradients[-1, :] = gradients[0, :]
    # if tileable[1]:
    #     gradients[:, -1] = gradients[:, 0]

    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[: -d[0], : -d[1]]
    g10 = gradients[d[0] :, : -d[1]]
    g01 = gradients[: -d[0], d[1] :]
    g11 = gradients[d[0] :, d[1] :]
    # Ramps
    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def render_noise2d():
    W = 80
    H = 50
    RH = 40
    F = 10
    rng = np.random.default_rng()

    shape = (H, W)
    res = (H // F, W // F)
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = (
        np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(
            1, 2, 0
        )
        % 1
    )
    angles = 2 * np.pi * rng.random((res[0] + 1, res[1] + 1))
    i = 0
    while True:
        # Gradients

        if i % d[0] == 0:
            angles = np.roll(angles, 1, axis=0)
            angles[0] = 2 * np.pi * rng.random((res[1] + 1,))

            greyscale = ".,-~:;=!*#$@"
            noise = np.round(
                ((perlin_noise(angles, grid, d)[0] + 1) / 2)
                * (len(greyscale) - 1)
            ).astype(int)

        noise = np.roll(noise, 1, axis=0)

        chars = "|\n".join(
            ["_" * W]
            + [
                "".join([greyscale[noise[j, i]] for i in range(W)])
                for j in range(H - RH, H)
            ]
            + ["_" * W]
        )
        # print("\x1b[H")
        print(chars)
        i += 1
        time.sleep(0.5)


def perlin_normals(noise):
    dx_y = -np.hstack((noise[:, 1:], noise[:, [-1]])) + np.hstack(
        (noise[:, [0]], noise[:, :-1])
    )
    dx = np.empty((*dx_y.shape, 3))
    dx[..., 0] = 1
    dx[..., 1] = dx_y
    dx[..., 2] = 0

    dz_y = -np.vstack((noise[1:, :], noise[[-1], :])) + np.vstack(
        (noise[[0], :], noise[:-1, :])
    )
    dz = np.empty((*dz_y.shape, 3))
    dz[..., 0] = 0
    dz[..., 1] = dz_y
    dz[..., 2] = 1
    normals = -np.cross(dz, dx)
    return normals


def render_noise3d():
    W = 60
    H = 90
    F = 30  # perlin resolution factor
    S = 4  # spatial resolution
    A = 15  # altitude

    pos = np.array([0, -10, W])

    light_dir = np.array([-0.6, -1, 1])
    light_dir = light_dir / np.linalg.norm(light_dir)
    # light_dir *= 0.9

    rot = Rotation.from_euler("xyz", [-10, 0, 0], degrees=True).as_matrix()

    rng = np.random.default_rng()

    hh = np.linspace(-H, H, S * H)
    ww = np.linspace(-W, W, S * W)
    g = np.array(np.meshgrid(hh, ww)).swapaxes(0, 2)

    PH = H * S
    PW = W * S

    shape = (PH, PW)
    res = (PH // F, PW // F)
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = (
        np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(
            1, 2, 0
        )
        % 1
    )
    angles = 2 * np.pi * rng.random((res[0] + 1, res[1] + 1))

    i = 0
    while True:
        # Gradients

        t = time.time()

        if i % d[0] == 0:
            angles = np.roll(angles, -1, axis=1)
            angles[0] = 2 * np.pi * rng.random((res[1] + 1,))

            greyscale = ".,-~:;=!*#$@"
            noise = ((perlin_noise(angles, grid, d) + 1) / 2) * A
            points = np.dstack(
                (g[..., [0]], noise[..., np.newaxis], g[..., [1]])
            )
            normals = perlin_normals(noise)

        noise = np.roll(noise, -1, axis=1)
        points = np.dstack((g[..., [0]], noise[..., np.newaxis], g[..., [1]]))
        normals = np.roll(normals, -1, axis=1)

        render(
            (points[:, :-F].reshape(-1, 3), normals[:, :-F].reshape(-1, 3)),
            rot,
            pos,
            light_dir,
        )

        i += 1
        print("%.2f fps" % (1 / (time.time() - t)))
        time.sleep(1 / 30)


def main():
    # render_cube()
    # render_sins()
    # render_noise2d()
    render_noise3d()


if __name__ == "__main__":
    main()
